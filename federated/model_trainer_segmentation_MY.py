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
        
        softmax_pred = torch.nn.functional.softmax(pred, dim=1)
        seg_pred = torch.argmax(softmax_pred, dim=1)
        
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
        if gt.shape[2:] != pred.shape[2:]:
            gt = F.interpolate(gt.float(), size=pred.shape[2:], mode='nearest')

        # print(f"Forward - Resized Ground Truth shape: {gt.shape}")

        gt = gt.squeeze(1).long() 
        num_classes = pred.shape[1]
        gt_one_hot = F.one_hot(gt, num_classes=num_classes)  # [batch_size, height, width, num_classes]
        gt_one_hot = gt_one_hot.permute(0, 3, 1, 2).float()  # [batch_size, num_classes, height, width]

        if self.activation == 'softmax':
            pred = F.softmax(pred, dim=1)
        elif self.activation == 'sigmoid':
            pred = torch.sigmoid(pred)

        loss = 0
        for i in range(num_classes):
            intersect = torch.sum(pred[:, i, ...] * gt_one_hot[:, i, ...])
            z_sum = torch.sum(pred[:, i, ...])
            y_sum = torch.sum(gt_one_hot[:, i, ...])
            loss += (2 * intersect + self.smooth) / (z_sum + y_sum + self.smooth)


        loss = 1 - loss / num_classes
        return loss

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
        x = x.to(dtype=self.model[0].weight.dtype)
        return self.model(x)
    
class ModelTrainerSegmentation(ModelTrainer):
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

        update_var_list = []
        names = generate_names()
        for name, param in model_adapt.named_parameters():
            param.requires_grad_(False)
            if "bn" in name or name in names:
                param.requires_grad_(True)
                update_var_list.append(param)

        optimizer = optim.Adam(update_var_list, lr=1e-3, betas=(0.9, 0.999))
        criterion = DiceLoss().to(device)

        def activate_dropout(m):
            if isinstance(m, nn.Dropout):
                m.train()
        model_adapt.apply(activate_dropout)

        # Maintain snapshots for Online Model Ensembling
        model_snapshots = deque(maxlen=3)
        model_snapshots.append(copy.deepcopy(model_adapt))
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 256, 256).to(device)  
            dummy_output = model_adapt(dummy_input)
            num_classes = dummy_output.shape[1]  
        discriminator = Discriminator(input_channels=num_classes).to(device)
        disc_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
        
        for epoch in range(10):
            loss_all = 0
            test_acc = 0.
            
            adv_weight = min(epoch / 10, 1.0)  
    
            for step, (data, target) in enumerate(test_data):
                data = data.to(device)
                target = target.to(device)
                
                if target.dim() == 3:
                    target = target.unsqueeze(1)

                with torch.cuda.amp.autocast():
                    output = model_adapt(data) 

                num_classes = output.shape[1]
                target_one_hot = F.one_hot(target.squeeze(1).long(), num_classes=num_classes)  # [B, H, W, C]
                target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

                real_outputs = discriminator(target_one_hot) 
                fake_outputs = discriminator(output.detach()) 

                real_labels = torch.ones_like(real_outputs, device=device)
                fake_labels = torch.zeros_like(fake_outputs, device=device)

                real_loss = F.binary_cross_entropy(real_outputs, real_labels)
                fake_loss = F.binary_cross_entropy(fake_outputs, fake_labels)

                disc_loss = (real_loss + fake_loss) / 2
                disc_loss.backward()
                disc_optimizer.step()
                
                adv_loss = F.binary_cross_entropy(discriminator(output), real_labels)  
                # Create noise-augmented data
                input_u1, input_u2 = data.clone(), data.clone()
                input_u1 = transforms_for_noise(input_u1, 0.5)
                input_u1, rot_mask, flip_mask = transforms_for_rot(input_u1)

                with torch.cuda.amp.autocast():
                    output = model_adapt(data)
                    loss_entropy_before = entropy_loss(output, c=2)

                    # Boundary Loss
                    boundary_loss_value = boundary_loss(output, target)

                    # Weighted Cross Entropy Loss
                    weights = torch.tensor([1.0, 5.0], device=device)
                    target_resized = F.interpolate(target.float(), size=output.shape[2:], mode='nearest').squeeze(1).long()
                    weighted_ce = weighted_cross_entropy_loss(output, target_resized, weights)

                discriminator.train()
                disc_optimizer.zero_grad()

                all_loss = (
                    15 * loss_entropy_before +
                    20.0 * boundary_loss_value +
                    0.2 * weighted_ce +
                    1 * adv_loss  
                )

                if epoch > 2:
                    pseudo_output = model_adapt(data)
                    pseudo_probs = torch.softmax(pseudo_output, dim=1)
                    pseudo_confidence, pseudo_target = torch.max(pseudo_probs, dim=1)

                    confidence_threshold = max(0.8 - 0.3 * (epoch / 10), 0.5)  
                    mask = pseudo_confidence > confidence_threshold
                    pseudo_target = torch.where(mask, pseudo_target, torch.zeros_like(pseudo_target))
                    pseudo_target = pseudo_target.unsqueeze(1)

                    pseudo_weight = torch.mean(pseudo_confidence)  

                    pseudo_loss = criterion(pseudo_output, pseudo_target)
                    all_loss += pseudo_weight * pseudo_loss 

                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()

                del input_u1, input_u2, rot_mask, flip_mask
                if epoch > 2:
                    del pseudo_output, pseudo_probs
                torch.cuda.empty_cache()

                with torch.no_grad():
                    loss_all += all_loss.item()
                    test_acc += DiceLoss().dice_coef(output.detach(), target).item()

            avg_acc = 0
            for _ in range(5):
                with torch.no_grad():
                    avg_output = model_adapt(data)
                    avg_acc += DiceLoss().dice_coef(avg_output.detach(), target).item()
            avg_acc /= 5

            model_snapshots.append(copy.deepcopy(model_adapt))
            ensemble_output = sum([snap(data) for snap in model_snapshots]) / len(model_snapshots)
            report_acc = round(DiceLoss().dice_coef(ensemble_output.detach(), target).item(), 4)

            print('Test Acc:', report_acc)
            if best_dice < report_acc:
                best_dice = report_acc
            dice_buffer.append(report_acc)
            print('Acc History:', dice_buffer)

        metrics['test_loss'] = loss_all / len(test_data)
        metrics['test_dice'] = best_dice
        print('Best Acc:', round(best_dice, 4))
        return metrics
    
    def train(self, train_data, device, args):
        model = self.model
        model.to(device)
        model.train()

        # train and update
        criterion = DiceLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr, amsgrad=True)

        epoch_loss = []
        epoch_acc = []
        for epoch in range(args.wk_iters):
            batch_loss = []
            batch_acc = []
            for batch_idx, (x, labels) in enumerate(train_data):
                model.zero_grad()
                x, labels = x.to(device), labels.to(device)
                log_probs = model(x)
                labels = F.interpolate(labels, size=log_probs.shape[2:], mode='nearest')
                
                loss = criterion(log_probs, labels)
                
                acc = DiceLoss().dice_coef(log_probs, labels).item()

                loss.backward()

                optimizer.step()
            
                batch_loss.append(loss.item())
                batch_acc.append(acc)
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_acc.append(sum(batch_acc) / len(batch_acc))
            logging.info('Client Index = {}\tEpoch: {}\tAcc:{:.4f}\tLoss: {:.4f}'.format(
                self.id, epoch, sum(epoch_acc) / len(epoch_acc),sum(epoch_loss) / len(epoch_loss)))
    def test(self, test_data, device, args, ood=False):
        model = copy.deepcopy(self.model)
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
        with torch.no_grad():
            # 对测试数据中的每个批次，执行以下操作：
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)
                acc = DiceLoss().dice_coef(pred, target).item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_acc'] += acc
        metrics["test_loss"] = metrics["test_loss"] / len(test_data)
        metrics["test_acc"] = metrics["test_acc"] / len(test_data)
        return metrics
    
def generate_names():
    names = []
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
    names.append('conv._routing_fn.fc1.weight')
    names.append('conv._routing_fn.fc1.bias')
    names.append('conv._routing_fn.fc2.weight')
    names.append('conv._routing_fn.fc2.bias')
    for i in range(1, 5):
        name = "upconv{}._routing_fn.fc1.weight".format(i)
        names.append(name)
        name = "upconv{}._routing_fn.fc1.bias".format(i)
        names.append(name)
        name = "upconv{}._routing_fn.fc2.weight".format(i)
        names.append(name)
        name = "upconv{}._routing_fn.fc2.bias".format(i)
        names.append(name)
    return names
