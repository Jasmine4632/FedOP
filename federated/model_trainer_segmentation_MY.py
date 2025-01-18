import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import logging
import cv2
import torch
from torch.cuda.amp import autocast, GradScaler
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from .model_trainer import ModelTrainer
from utils.loss import entropy_loss
from data.prostate.transforms import transforms_for_noise, transforms_for_rot, transforms_for_scale, transforms_back_scale, transforms_back_rot
import copy
import numpy as np
import random
import torch.optim as optim
from torch.nn.functional import dropout
from collections import deque
from torchvision import transforms as T

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, activation='sigmoid'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.activation = activation

    def dice_coef(self, pred, gt):
        """Computational formula for Dice coefficient"""
        
        # Ensure ground truth matches the prediction dimensions
        if gt.shape[2:] != pred.shape[2:]:
            gt = F.interpolate(gt.float(), size=pred.shape[2:], mode='nearest')
        
        # 使用 softmax 获取每个类别的概率
        softmax_pred = torch.nn.functional.softmax(pred, dim=1)
        seg_pred = torch.argmax(softmax_pred, dim=1)
        
        # 去掉标签的通道维度
        gt = gt.squeeze(1)  # Remove the channel dimension of gt (from [B, 1, H, W] to [B, H, W])

        all_dice = 0
        batch_size = gt.shape[0]
        num_classes = softmax_pred.shape[1]

        for i in range(num_classes):
            each_pred = (seg_pred == i).float()  # Use direct comparison to get binary mask for each class
            each_gt = (gt == i).float()

            intersection = torch.sum((each_pred * each_gt).view(batch_size, -1), dim=1)
            union = each_pred.view(batch_size, -1).sum(1) + each_gt.view(batch_size, -1).sum(1)
            dice = (2. * intersection + self.smooth) / (union + self.smooth)

            all_dice += torch.mean(dice)

        return all_dice / num_classes

    def forward(self, pred, gt):
        
        # 确保标签和预测在所有维度上匹配
        if gt.shape[2:] != pred.shape[2:]:
            gt = F.interpolate(gt.float(), size=pred.shape[2:], mode='nearest')

        # # 调试输出调整后的标签形状
        # print(f"Forward - Resized Ground Truth shape: {gt.shape}")

        # 转换标签为 one-hot 编码
        gt = gt.squeeze(1).long()  # 去掉标签的通道维度并转为 long 类型
        num_classes = pred.shape[1]
        gt_one_hot = F.one_hot(gt, num_classes=num_classes)  # [batch_size, height, width, num_classes]
        gt_one_hot = gt_one_hot.permute(0, 3, 1, 2).float()  # [batch_size, num_classes, height, width]

        # 激活函数
        if self.activation == 'softmax':
            pred = F.softmax(pred, dim=1)
        elif self.activation == 'sigmoid':
            pred = torch.sigmoid(pred)

        # 计算损失
        loss = 0
        for i in range(num_classes):
            intersect = torch.sum(pred[:, i, ...] * gt_one_hot[:, i, ...])
            z_sum = torch.sum(pred[:, i, ...])
            y_sum = torch.sum(gt_one_hot[:, i, ...])
            loss += (2 * intersect + self.smooth) / (z_sum + y_sum + self.smooth)


        loss = 1 - loss / num_classes

        # # 调试输出最终损失值
        # print(f"Forward - Dice Loss: {loss.item()}")

        return loss

# 计算输入和目标logits的Softmax均方误差损失
def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss
    
def deterministic(seed):
     cudnn.benchmark = False
     cudnn.deterministic = True
     np.random.seed(seed)
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     random.seed(seed)
    
def boundary_loss(pred, gt):
    """
    Calculate boundary loss to penalize incorrect predictions on the boundary of objects.
    Args:
        pred (torch.Tensor): Predicted logits of shape (batch_size, num_classes, H, W)
        gt (torch.Tensor): Ground truth labels of shape (batch_size, 1, H, W)

    Returns:
        torch.Tensor: Scalar boundary loss
    """
    # Ensure predictions and ground truth have the same size
    if pred.shape[2:] != gt.shape[2:]:
        gt = F.interpolate(gt.float(), size=pred.shape[2:], mode='nearest')

    # Convert logits to probabilities
    pred_probs = torch.softmax(pred, dim=1)

    # Binary mask for each class (assuming binary segmentation)
    boundary_gt = (gt == 1).float()

    # Compute gradients of ground truth and predictions
    gt_grad = torch.nn.functional.conv2d(boundary_gt, weight=torch.ones((1, 1, 3, 3)).to(gt.device), padding=1)
    pred_grad = torch.nn.functional.conv2d(pred_probs[:, 1:2, :, :], weight=torch.ones((1, 1, 3, 3)).to(pred.device), padding=1)

    # Compute absolute difference of gradients
    boundary_diff = torch.abs(pred_grad - gt_grad)

    # Average the loss across the batch
    return boundary_diff.mean()

def weighted_cross_entropy_loss(pred, target, weights):
    """
    Compute weighted cross entropy loss.
    Args:
        pred (torch.Tensor): Predicted logits of shape (batch_size, num_classes, H, W).
        target (torch.Tensor): Ground truth labels of shape (batch_size, H, W).
        weights (list or torch.Tensor): Class weights, list of length num_classes.

    Returns:
        torch.Tensor: Weighted cross entropy loss.
    """
    if not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights, device=pred.device, dtype=torch.float32)

    # Ensure target is 2D (batch_size, H, W)
    if target.dim() == 4 and target.size(1) == 1:
        target = target.squeeze(1)
        
    # Ensure the spatial dimensions match
    if target.shape[1:] != pred.shape[2:]:
        raise ValueError(f"Shape mismatch: target {target.shape} and pred {pred.shape}")

    # Compute log probabilities
    log_probs = F.log_softmax(pred, dim=1)

    # Convert target to one-hot
    target_one_hot = F.one_hot(target.long(), num_classes=pred.size(1))  # [batch_size, H, W, num_classes]
    target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # [batch_size, num_classes, H, W]

    # Apply weights to the one-hot encoded targets
    weighted_target = target_one_hot * weights.view(1, -1, 1, 1)

    # Compute the loss
    loss = -torch.sum(weighted_target * log_probs, dim=1).mean()

    return loss

class Discriminator(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim * 4, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 确保 x 和模型参数类型一致
        x = x.to(dtype=self.model[0].weight.dtype)
        return self.model(x)
    
# 这个函数的主要目的是生成在模型自适应过程中需要更新的特定参数的名称列表。
# 在 io_pfl 方法中，使用这个名称列表来识别和更新模型中的特定参数，从而实现渐进式的联邦学习，改善模型在新任务或数据集上的表现。
class ModelTrainerSegmentation(ModelTrainer):
    # get_model_params 和 set_model_params 方法用于获取和设置模型参数
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)
    
    def io_pfl(self, test_data, device, args):
        deterministic(args.seed)
        metrics = {
            'test_dice': 0,
            'test_loss': 0,
        }
        best_dice = 0.
        dice_buffer = []

        # Current model and adaptive copy
        model = self.model
        model_adapt = copy.deepcopy(model)
        model_adapt = nn.DataParallel(model_adapt)  # Multi-GPU support
        model_adapt.to(device)
        model_adapt.train()

        # Adaptive Batch Normalization Setup
        for m in model_adapt.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

        # Parameter freezing and selective update setup
        update_var_list = []
        names = generate_names()
        for name, param in model_adapt.named_parameters():
            param.requires_grad_(False)
            if "bn" in name or name in names:
                param.requires_grad_(True)
                update_var_list.append(param)

        optimizer = optim.Adam(update_var_list, lr=1e-3, betas=(0.9, 0.999))
        criterion = DiceLoss().to(device)

        # Monte Carlo Dropout - activating Dropout during testing
        def activate_dropout(m):
            if isinstance(m, nn.Dropout):
                m.train()
        model_adapt.apply(activate_dropout)

        # Maintain snapshots for Online Model Ensembling
        model_snapshots = deque(maxlen=3)
        model_snapshots.append(copy.deepcopy(model_adapt))
        
        # 推断输出通道数
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 256, 256).to(device)  # 假设输入是 (B, C, H, W)，可以根据实际情况修改
            dummy_output = model_adapt(dummy_input)
            num_classes = dummy_output.shape[1]  # 输出通道数
        discriminator = Discriminator(input_channels=num_classes).to(device)
        disc_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
        
        # Begin adaptive training over 10 epochs
        for epoch in range(10):
            loss_all = 0
            test_acc = 0.
            
            # 动态调整对抗性损失权重
            adv_weight = min(epoch / 10, 1.0)  # 从0逐步增加
    
            for step, (data, target) in enumerate(test_data):
                data = data.to(device)
                target = target.to(device)
                
                # 确保 target 是 [batch_size, 1, height, width]
                if target.dim() == 3:
                    target = target.unsqueeze(1)

                with torch.cuda.amp.autocast():
                    output = model_adapt(data)  # 模型的预测输出

                # 转换 target 为 one-hot 编码
                num_classes = output.shape[1]
                target_one_hot = F.one_hot(target.squeeze(1).long(), num_classes=num_classes)  # [B, H, W, C]
                target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

                # 判别器训练阶段
                real_outputs = discriminator(target_one_hot)  # 真实目标
                fake_outputs = discriminator(output.detach())  # 预测目标

                real_labels = torch.ones_like(real_outputs, device=device)
                fake_labels = torch.zeros_like(fake_outputs, device=device)

                real_loss = F.binary_cross_entropy(real_outputs, real_labels)
                fake_loss = F.binary_cross_entropy(fake_outputs, fake_labels)

                # 判别器的总损失
                disc_loss = (real_loss + fake_loss) / 2
                disc_loss.backward()
                disc_optimizer.step()
                
                # 生成器（分割模型）的训练阶段
                adv_loss = F.binary_cross_entropy(discriminator(output), real_labels)  # 生成器试图“骗过”判别器
                # Create noise-augmented data
                input_u1, input_u2 = data.clone(), data.clone()
                input_u1 = transforms_for_noise(input_u1, 0.5)
                input_u1, rot_mask, flip_mask = transforms_for_rot(input_u1)

                # Forward pass
                with torch.cuda.amp.autocast():
                    output = model_adapt(data)
                    loss_entropy_before = entropy_loss(output, c=2)

                    # Boundary Loss
                    boundary_loss_value = boundary_loss(output, target)

                    # Weighted Cross Entropy Loss
                    weights = torch.tensor([1.0, 5.0], device=device)
                    target_resized = F.interpolate(target.float(), size=output.shape[2:], mode='nearest').squeeze(1).long()
                    weighted_ce = weighted_cross_entropy_loss(output, target_resized, weights)

                # 判别器训练阶段
                discriminator.train()
                disc_optimizer.zero_grad()

                all_loss = (
                    15 * loss_entropy_before +
                    20.0 * boundary_loss_value +
                    0.2 * weighted_ce +
                    1 * adv_loss  # 动态调整对抗性损失权重
                )

                if epoch > 2:
                    pseudo_output = model_adapt(data)
                    pseudo_probs = torch.softmax(pseudo_output, dim=1)
                    pseudo_confidence, pseudo_target = torch.max(pseudo_probs, dim=1)

                    # 动态调整置信度阈值
                    confidence_threshold = max(0.8 - 0.3 * (epoch / 10), 0.5)  # 阈值从 0.8 动态减少到 0.5
                    mask = pseudo_confidence > confidence_threshold
                    pseudo_target = torch.where(mask, pseudo_target, torch.zeros_like(pseudo_target))
                    pseudo_target = pseudo_target.unsqueeze(1)

                    # 动态伪标签权重
                    pseudo_weight = torch.mean(pseudo_confidence)  # 取置信度均值作为权重

                    # 结合伪标签损失、边界正则化和一致性正则化
                    pseudo_loss = criterion(pseudo_output, pseudo_target)
                    all_loss += pseudo_weight * pseudo_loss  # 伪标签损失

                # Backpropagation and optimization
                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()

                # Clear unused variables and cache
                del input_u1, input_u2, rot_mask, flip_mask
                if epoch > 2:
                    del pseudo_output, pseudo_probs
                torch.cuda.empty_cache()

                # Update metrics
                with torch.no_grad():
                    loss_all += all_loss.item()
                    test_acc += DiceLoss().dice_coef(output.detach(), target).item()

            # Monte Carlo Dropout - averaging multiple predictions for robustness
            avg_acc = 0
            for _ in range(5):
                with torch.no_grad():
                    avg_output = model_adapt(data)
                    avg_acc += DiceLoss().dice_coef(avg_output.detach(), target).item()
            avg_acc /= 5

            # Ensemble Model Updating
            model_snapshots.append(copy.deepcopy(model_adapt))
            ensemble_output = sum([snap(data) for snap in model_snapshots]) / len(model_snapshots)
            report_acc = round(DiceLoss().dice_coef(ensemble_output.detach(), target).item(), 4)

            print('Test Acc:', report_acc)
            if best_dice < report_acc:
                best_dice = report_acc
            dice_buffer.append(report_acc)
            print('Acc History:', dice_buffer)

        # Final metrics
        metrics['test_loss'] = loss_all / len(test_data)
        metrics['test_dice'] = best_dice
        print('Best Acc:', round(best_dice, 4))
        return metrics
    
#     @torch.enable_grad()
#     # io_pfl 方法进行测试时间自适应的渐进联邦学习（PFL）
#     def io_pfl(self, test_data, device, args):
#         deterministic(args.seed)
#         metrics = {
#             'test_dice': 0,
#             'test_loss': 0,
#         }
#         best_dice = 0.
#         dice_buffer = []
        
#         # Current model and adaptive copy
#         model = self.model
#         model_adapt = copy.deepcopy(model)
#         model_adapt.to(device)
#         model_adapt.train()
        
#         # Adaptive Batch Normalization Setup
#         for m in model_adapt.modules():
#             if isinstance(m, nn.BatchNorm2d):
#                 m.requires_grad_(True)
#                 m.track_running_stats = False
#                 m.running_mean = None
#                 m.running_var = None
                
#         # Parameter freezing and selective update setup
#         update_var_list = []
#         update_name_list = []
#         names = generate_names()
        
#         for name, param in model_adapt.named_parameters():
#             param.requires_grad_(False)
#             if "bn" in name or name in names:
#                 param.requires_grad_(True)
#                 update_var_list.append(param)
#                 update_name_list.append(name)
        
#         optimizer = optim.Adam(update_var_list, lr=1e-3, betas=(0.9, 0.999))
#         criterion = DiceLoss().to(device)
#         loss_all = 0
#         test_acc = 0.
        
#         # Monte Carlo Dropout - activating Dropout during testing
#         def activate_dropout(m):
#             if type(m) == nn.Dropout:
#                 m.train()
#         model_adapt.apply(activate_dropout)
        
#         # Maintain snapshots for Online Model Ensembling
#         model_snapshots = deque(maxlen=3)
#         model_snapshots.append(copy.deepcopy(model_adapt))

#         # Begin adaptive training over 10 epochs
#         for epoch in range(10):
#             loss_all = 0
#             test_acc = 0.
#             deterministic(args.seed)
#             for step, (data, target) in enumerate(test_data):
#                 deterministic(args.seed)
#                 data = data.to(device)
#                 target = target.to(device)

#                 # Create noise augmented data if needed
#                 if epoch > -1:
#                     input_u1 = copy.deepcopy(data)
#                     input_u2 = copy.deepcopy(data)
#                     input_u1 = transforms_for_noise(input_u1, 0.5)
#                     input_u1, rot_mask, flip_mask = transforms_for_rot(input_u1)
                
#                 # Forward pass with model
#                 output = model_adapt(data)
#                 loss_entropy_before = entropy_loss(output, c=2)
                
#                 if epoch > -1:
#                     output_u1 = model_adapt(input_u1)
#                     output_u2 = model_adapt(input_u2)
#                     output_u1 = transforms_back_rot(output_u1, rot_mask, flip_mask)
#                     consistency_loss = softmax_mse_loss(output_u1, output_u2)
                
#                 # Smoothness loss
#                 loss_smooth = smooth_loss(output)
#                 loss_smooth = torch.norm(loss_smooth)
                
#                 # Boundary Loss
#                 boundary_loss_value = boundary_loss(output, target)

#                 # Shape Regularization Loss
#                 shape_loss = shape_regularization_loss(output)

#                 # Weighted Cross Entropy Loss
#                 weights = torch.tensor([1.0, 5.0], device=device)
#                 # Compute weighted cross-entropy loss
#                 # Ensure target shape matches output spatial dimensions
#                 if target.dim() == 4 and target.shape[1] == 1:  # [N, 1, H, W]
#                     target_resized = F.interpolate(target.float(), size=output.shape[2:], mode='nearest').squeeze(1).long()  # [N, H', W']
#                 elif target.dim() == 3:  # [N, H, W]
#                     target_resized = F.interpolate(target.unsqueeze(1).float(), size=output.shape[2:], mode='nearest').squeeze(1).long()
#                 else:
#                     raise ValueError(f"Unexpected target shape: {target.shape}")

#                 # Calculate weighted cross-entropy loss
#                 weighted_ce = weighted_cross_entropy_loss(output, target_resized, weights)

#                 # Total Loss Calculation
#                 all_loss = (
#                     6 * loss_entropy_before +
#                     0.25 * torch.mean(consistency_loss) +
#                     0.5 * loss_smooth +
#                     10.0 * boundary_loss_value +
#                     0.05 * shape_loss +
#                     0.2 * weighted_ce
#                 )
# #             0.3 0.1 0.3 Acc History: [0.893, 0.8983, 0.871, 0.8823, 0.9004, 0.9023, 0.903, 0.9034, 0.9043, 0.9053]
#                 # # Total loss calculation
#                 # all_loss = 6 * loss_entropy_before + 0.25 * torch.mean(consistency_loss) + 0.5 * loss_smooth #0.8812

#                 if epoch > 2:  # Start using pseudo-label loss after 2 epochs
#                     pseudo_output = model_adapt(data)
#                     pseudo_probs = torch.softmax(pseudo_output, dim=1)
#                     pseudo_confidence, pseudo_target = torch.max(pseudo_probs, dim=1)
                    
#                     # Dynamically adjust the confidence threshold
#                     confidence_threshold = 0.7  # You can decrease it as needed
#                     if epoch > 5:  # After a few epochs, lower the threshold
#                         confidence_threshold = 0.5
                    
#                     mask = pseudo_confidence > confidence_threshold
#                     pseudo_target = torch.where(mask, pseudo_target, torch.zeros_like(pseudo_target))

#                     pseudo_target = pseudo_target.unsqueeze(1)  # (batch_size, 1, 96, 96)
#                     pseudo_loss = criterion(pseudo_output, pseudo_target)
#                     pseudo_weight = 0.2  # Initial weight for pseudo-labels
                    
#                     # Gradually increase the weight of pseudo-labels as training progresses
#                     if epoch > 5:
#                         pseudo_weight = 2  # Increase weight after a few epochs
                    
#                     all_loss += pseudo_weight * pseudo_loss  # Add pseudo-label loss
#                 # Backpropagation and optimization
#                 optimizer.zero_grad()
#                 all_loss.backward()
#                 optimizer.step()

#                 # Log pseudo loss
#                 # Re-calculate metrics after optimization
#                 output = model_adapt(data)
#                 loss = criterion(output, target)
#                 loss_all += loss.item()
#                 test_acc += DiceLoss().dice_coef(output, target).item()

#             # Monte Carlo Dropout - averaging multiple predictions for robustness
#             avg_acc = 0
#             for _ in range(5):  # Performing multiple predictions with Dropout enabled
#                 avg_output = model_adapt(data)
#                 avg_acc += DiceLoss().dice_coef(avg_output, target).item()
#             avg_acc /= 5

#             # Ensemble Model Updating
#             model_snapshots.append(copy.deepcopy(model_adapt))
#             ensemble_output = sum([snap(data) for snap in model_snapshots]) / len(model_snapshots)
#             report_acc = round(DiceLoss().dice_coef(ensemble_output, target).item(), 4)
            
#             print('Test Acc:', report_acc)
#             if best_dice < report_acc:
#                 best_dice = report_acc
#             dice_buffer.append(report_acc)
#             print('Acc History:', dice_buffer)

#         # Final metrics
#         loss = loss_all / len(test_data)
#         acc = best_dice
#         print('Best Acc:', round(best_dice, 4))
#         metrics['test_loss'] = loss
#         metrics['test_dice'] = acc
#         return metrics
    
    #  train 方法进行模型训练
    def train(self, train_data, device, args):
        model = self.model
        model.to(device)
        model.train()

        # train and update
        # 损失函数使用自定义的 DiceLoss，这是一个常用于分割任务的损失函数，评估模型输出和真实标签的 Dice 系数
        criterion = DiceLoss().to(device)
        # 优化器可以根据输入参数 args.client_optimizer 的值选择使用 SGD（随机梯度下降）或 Adam（自适应动量估计）
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr, amsgrad=True)

        # 按照指定的迭代次数（args.wk_iters），对每个数据批次进行训练。
        # 对于每个批次的数据，将输入数据 x 和标签 labels 转移到设备上，然后通过模型进行前向传播得到预测概率 log_probs。
        # 计算预测与真实标签之间的损失 loss，并通过 loss.backward() 进行反向传播，计算每个参数的梯度。
        # 使用优化器的 optimizer.step() 方法更新模型参数。
        epoch_loss = []
        epoch_acc = []
        for epoch in range(args.wk_iters):
            batch_loss = []
            batch_acc = []
            for batch_idx, (x, labels) in enumerate(train_data):
                model.zero_grad()
                x, labels = x.to(device), labels.to(device)
                log_probs = model(x)
                # 调整标签尺寸，使其与模型输出匹配
                labels = F.interpolate(labels, size=log_probs.shape[2:], mode='nearest')
                
                loss = criterion(log_probs, labels)
                
                acc = DiceLoss().dice_coef(log_probs, labels).item()

                loss.backward()

                optimizer.step()
            
                batch_loss.append(loss.item())
                batch_acc.append(acc)
            # 记录每个批次的损失和准确率，并计算每个 epoch 的平均损失和准确率。
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_acc.append(sum(batch_acc) / len(batch_acc))
            # 使用 logging.info 记录每个 epoch 的平均损失和准确率，方便后续分析和调试
            logging.info('Client Index = {}\tEpoch: {}\tAcc:{:.4f}\tLoss: {:.4f}'.format(
                self.id, epoch, sum(epoch_acc) / len(epoch_acc),sum(epoch_loss) / len(epoch_loss)))
    # test 方法进行模型测试
    def test(self, test_data, device, args, ood=False):
        model = copy.deepcopy(self.model)
        # 将模型的副本移至指定设备，并根据是否为 ood（out-of-distribution，分布外测试）设置模型为训练模式或评估模式
        model.to(device)
        if ood:
            model.train()
        else:
            model.eval()

        metrics = {
            'test_acc': 0,
            'test_loss': 0,
        }

        criterion = DiceLoss().to(device)
        # 不需要计算梯度，因此使用 torch.no_grad() 上下文管理器，禁用梯度计算以节省内存和加速。
        with torch.no_grad():
            # 对测试数据中的每个批次，执行以下操作：
            for batch_idx, (x, target) in enumerate(test_data):
                # 将输入数据和目标标签转移到设备上
                x = x.to(device)
                target = target.to(device)
                # 使用模型进行前向传播，得到预测值 pred。
                pred = model(x)
                # 使用 DiceLoss 计算预测值与真实标签之间的损失。
                loss = criterion(pred, target)
                # 计算预测的 Dice 系数（分割任务的准确率）。
                acc = DiceLoss().dice_coef(pred, target).item()
                # 累加每个批次的损失和准确率，用于计算平均值
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_acc'] += acc
        metrics["test_loss"] = metrics["test_loss"] / len(test_data)
        metrics["test_acc"] = metrics["test_acc"] / len(test_data)
        return metrics
    
# generate_names 函数生成编码器和解码器中的路由层参数名称列表
def generate_names():
    # 初始化名称列表
    names = []
    # 生成编码器和解码器的路由层参数名称
    # 这个循环生成四个编码器（encoder1 到 encoder4）和四个解码器（decoder1 到 decoder4）中两个卷积层（conv1 和 conv2）的路由层参数名称，包括全连接层（fc1 和 fc2）的权重和偏置
    for i in range(1, 5):
        for j in range(1, 3):
            name = "encoder{}.enc{}_conv{}._routing_fn.fc1.weight".format(i,i,j)
            names.append(name)
            name = "encoder{}.enc{}_conv{}._routing_fn.fc1.bias".format(i,i,j)
            names.append(name)
            name = "decoder{}.dec{}_conv{}._routing_fn.fc1.weight".format(i,i,j)
            names.append(name)
            name = "decoder{}.dec{}_conv{}._routing_fn.fc1.bias".format(i,i,j)
            names.append(name)
            name = "encoder{}.enc{}_conv{}._routing_fn.fc2.weight".format(i,i,j)
            names.append(name)
            name = "encoder{}.enc{}_conv{}._routing_fn.fc2.bias".format(i,i,j)
            names.append(name)
            name = "decoder{}.dec{}_conv{}._routing_fn.fc2.weight".format(i,i,j)
            names.append(name)
            name = "decoder{}.dec{}_conv{}._routing_fn.fc2.bias".format(i,i,j)
            names.append(name)
    # 添加卷积层的路由层参数名称
    names.append('conv._routing_fn.fc1.weight')
    names.append('conv._routing_fn.fc1.bias')
    names.append('conv._routing_fn.fc2.weight')
    names.append('conv._routing_fn.fc2.bias')
    # 生成上采样卷积层的路由层参数名称
    # 这个循环生成四个上采样卷积层（upconv1 到 upconv4）的路由层参数名称，包括全连接层（fc1 和 fc2）的权重和偏置
    for i in range(1, 5):
        name = "upconv{}._routing_fn.fc1.weight".format(i)
        names.append(name)
        name = "upconv{}._routing_fn.fc1.bias".format(i)
        names.append(name)
        name = "upconv{}._routing_fn.fc2.weight".format(i)
        names.append(name)
        name = "upconv{}._routing_fn.fc2.bias".format(i)
        names.append(name)
    # 返回名称列表
    return names
