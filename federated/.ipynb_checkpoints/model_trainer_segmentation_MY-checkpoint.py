import logging
import cv2
import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from .model_trainer import ModelTrainer
import copy
import numpy as np
import random


def entropy_loss(p, c=3):
    # p N*C*W*H*D
    p = F.softmax(p, dim=1)
    y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=0) / torch.tensor(np.log(c)).cuda()
    ent = torch.mean(y1)
    return ent


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, activation='sigmoid'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.activation = activation

    def dice_coef(self, pred, gt):
        """ computational formula
        """

        softmax_pred = torch.nn.functional.softmax(pred, dim=1)
        seg_pred = torch.argmax(softmax_pred, dim=1)
        all_dice = 0
        gt = gt.squeeze(dim=1)
        batch_size = gt.shape[0]
        num_class = softmax_pred.shape[1]
        for i in range(num_class):
            each_pred = torch.zeros_like(seg_pred)
            each_pred[seg_pred == i] = 1

            each_gt = torch.zeros_like(gt)
            each_gt[gt == i] = 1

            intersection = torch.sum((each_pred * each_gt).view(batch_size, -1), dim=1)

            union = each_pred.view(batch_size, -1).sum(1) + each_gt.view(batch_size, -1).sum(1)
            dice = (2. * intersection) / (union + 1e-5)

            all_dice += torch.mean(dice)

        return all_dice * 1.0 / num_class

    def forward(self, pred, gt):
        sigmoid_pred = F.softmax(pred, dim=1)

        batch_size = gt.shape[0]
        num_class = sigmoid_pred.shape[1]

        # conver label to one-hot
        bg = torch.zeros_like(gt)
        bg[gt == 0] = 1
        label1 = torch.zeros_like(gt)
        label1[gt == 1] = 1
        label2 = torch.zeros_like(gt)
        label2[gt == 2] = 1
        label = torch.cat([bg, label1, label2], dim=1)

        loss = 0
        smooth = 1e-5

        for i in range(num_class):
            intersect = torch.sum(sigmoid_pred[:, i, ...] * label[:, i, ...])
            z_sum = torch.sum(sigmoid_pred[:, i, ...])
            y_sum = torch.sum(label[:, i, ...])
            loss += (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss * 1.0 / num_class
        return loss


def transforms_back_rot(ema_output, rot_mask, flip_mask):
    for idx in range(ema_output.shape[0]):

        ema_output[idx] = torch.rot90(ema_output[idx], int(rot_mask[idx]), dims=[2, 1])

        if flip_mask[idx] == 1:
            ema_output[idx] = torch.flip(ema_output[idx], [1])

    return ema_output


def transforms_for_noise(inputs_u2, std):
    gaussian = np.random.normal(0, std, (inputs_u2.shape[0], 3, inputs_u2.shape[-1], inputs_u2.shape[-1]))
    gaussian = torch.from_numpy(gaussian).float().cuda()
    inputs_u2_noise = inputs_u2 + gaussian

    return inputs_u2_noise


def transforms_for_rot(ema_inputs):
    rot_mask = np.random.randint(0, 4, ema_inputs.shape[0])
    flip_mask = np.random.randint(0, 2, ema_inputs.shape[0])

    for idx in range(ema_inputs.shape[0]):
        if flip_mask[idx] == 1:
            ema_inputs[idx] = torch.flip(ema_inputs[idx], [1])

        ema_inputs[idx] = torch.rot90(ema_inputs[idx], int(rot_mask[idx]), dims=[1, 2])

    return ema_inputs, rot_mask, flip_mask


def transforms_for_scale(ema_inputs, image_size):
    scale_mask = np.random.uniform(low=0.9, high=1.1, size=ema_inputs.shape[0])
    scale_mask = scale_mask * image_size
    scale_mask = [int(item) for item in scale_mask]
    scale_mask = [item - 1 if item % 2 != 0 else item for item in scale_mask]
    half_size = int(image_size / 2)

    ema_outputs = torch.zeros_like(ema_inputs)

    for idx in range(ema_inputs.shape[0]):
        # to numpy
        img = np.transpose(ema_inputs[idx].cpu().numpy(), (1, 2, 0))
        # crop
        if scale_mask[idx] > image_size:

            new_img1 = np.expand_dims(np.pad(img[:, :, 0],
                                             (int((scale_mask[idx] - image_size) / 2),
                                              int((scale_mask[idx] - image_size) / 2)), 'edge'), axis=-1)
            new_img2 = np.expand_dims(np.pad(img[:, :, 1],
                                             (int((scale_mask[idx] - image_size) / 2),
                                              int((scale_mask[idx] - image_size) / 2)), 'edge'), axis=-1)
            new_img3 = np.expand_dims(np.pad(img[:, :, 2],
                                             (int((scale_mask[idx] - image_size) / 2),
                                              int((scale_mask[idx] - image_size) / 2)), 'edge'), axis=-1)
            new_img = np.concatenate([new_img1, new_img2, new_img3], axis=-1)
            img = new_img
        else:
            img = img[half_size - int(scale_mask[idx] / 2):half_size + int(scale_mask[idx] / 2),
                  half_size - int(scale_mask[idx] / 2): half_size + int(scale_mask[idx] / 2), :]

        # resize
        resized_img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        # to tensor
        ema_outputs[idx] = torch.from_numpy(resized_img.transpose((2, 0, 1))).cuda()

    return ema_outputs, scale_mask


def transforms_back_scale(ema_inputs, scale_mask, image_size):
    half_size = int(image_size / 2)
    returned_img = np.zeros((ema_inputs.shape[0], image_size, image_size, 2))

    ema_outputs = torch.zeros_like(ema_inputs)

    for idx in range(ema_inputs.shape[0]):
        # to numpy
        img = np.transpose(ema_inputs[idx].cpu().detach().numpy(), (1, 2, 0))
        # resize
        resized_img = cv2.resize(img, (int(scale_mask[idx]), int(scale_mask[idx])), interpolation=cv2.INTER_CUBIC)

        if scale_mask[idx] > image_size:
            returned_img[idx] = resized_img[int(scale_mask[idx] / 2) - half_size:int(scale_mask[idx] / 2) + half_size,
                                int(scale_mask[idx] / 2) - half_size:int(scale_mask[idx] / 2) + half_size, :]

        else:
            returned_img[idx, half_size - int(scale_mask[idx] / 2):half_size + int(scale_mask[idx] / 2),
            half_size - int(scale_mask[idx] / 2): half_size + int(scale_mask[idx] / 2), :] = resized_img
        # to tensor
        ema_outputs[idx] = torch.from_numpy(returned_img[idx].transpose((2, 0, 1))).cuda()

    return ema_outputs, scale_mask


def postprocess_scale(input, scale_mask, image_size):
    half_size = int(input.shape[-1] / 2)
    new_input = torch.zeros((input.shape[0], 2, input.shape[-1], input.shape[-1]))

    for idx in range(input.shape[0]):

        if scale_mask[idx] > image_size:
            new_input = input

        else:
            new_input[idx, :, half_size - int(scale_mask[idx] / 2):half_size + int(scale_mask[idx] / 2),
            half_size - int(scale_mask[idx] / 2): half_size + int(scale_mask[idx] / 2)] \
                = input[idx, :, half_size - int(scale_mask[idx] / 2):half_size + int(scale_mask[idx] / 2),
                  half_size - int(scale_mask[idx] / 2): half_size + int(scale_mask[idx] / 2)]

    return new_input.cuda()


# 计算输出的平滑损失
def smooth_loss(output, d=10):
    
    output_pred = torch.nn.functional.softmax(output, dim=1)
    output_pred_foreground = output_pred[:,1:,:,:]
    m = nn.MaxPool2d(kernel_size=2*d+1, stride=1, padding=d)
    loss = (m(output_pred_foreground) + m(-output_pred_foreground))*(1e-3*1e-3)
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


# 这个函数的主要目的是生成在模型自适应过程中需要更新的特定参数的名称列表。
# 在 io_pfl 方法中，使用这个名称列表来识别和更新模型中的特定参数，从而实现渐进式的联邦学习，改善模型在新任务或数据集上的表现。
class ModelTrainerSegmentation(ModelTrainer):
    # get_model_params 和 set_model_params 方法用于获取和设置模型参数
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)
    
    @torch.enable_grad()
    # io_pfl 方法进行测试时间自适应的渐进联邦学习（PFL）
    def io_pfl(self, test_data, device, args):
        deterministic(args.seed)
        # 初始化 metrics 字典用于存储测试结果的 Dice 系数和损失     什么是test_dice？
        metrics = {
            'test_dice': 0,
            'test_loss': 0,
        }
        # 初始化 best_dice 变量用于记录最佳 Dice 系数
        best_dice = 0.
        # 初始化 dice_buffer 列表用于记录每轮的 Dice 系数
        dice_buffer = []
        # 获取当前模型 self.model
        model = self.model
        # 并复制为 model_adapt，用于自适应调整
        model_adapt = copy.deepcopy(model)
        # 将 model_adapt 移动到指定设备（如 GPU）
        model_adapt.to(device)
        # 设置 model_adapt 为训练模式                ！！! ！！！！！！train
        model_adapt.train()
        # 遍历 model_adapt 中的所有模块
        for m in model_adapt.modules():
            # 对于所有的 BatchNorm 层，设置其 requires_grad 为 True，使其参数在训练中更新
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # 禁用 BatchNorm 层的运行统计，并将其均值和方差设为 None
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
        #  获取 model_adapt 中所有参数的名字和参数列表
        var_list = model_adapt.named_parameters()
        # 初始化 update_var_list 和 update_name_list 列表，用于存储需要更新的参数和对应的名字
        update_var_list = []
        update_name_list = []
        # 调用 generate_names 函数生成包含路由层参数名称的列表
        names = generate_names()

        # 遍历 var_list 中的所有参数
        for idx, (name, param) in  enumerate(var_list):
            # 将所有参数的 requires_grad 属性设为 False，禁止其更新
            param.requires_grad_(False)
            # 如果参数名称中包含 "bn" 或在 names 列表中，设置其 requires_grad 为 True，允许其更新，并将其添加到 update_var_list 和 update_name_list 中
            if "bn" in name or name in names:
                param.requires_grad_(True)
                update_var_list.append(param)
                update_name_list.append(name)
        # 使用 Adam 优化器，设置学习率为 1e-3，只更新 update_var_list 中的参数
        optimizer = torch.optim.Adam(update_var_list, lr=1e-3, betas=(0.9, 0.999))
        # 初始化 DiceLoss 损失函数，并将其移动到指定设备
        criterion = DiceLoss().to(device)
        # 初始化 loss_all 和 test_acc 变量用于记录损失和准确率
        loss_all = 0
        test_acc = 0.
        
        # 进行 10 轮自适应训练
        for epoch in range(10):
            # 每轮训练开始时，将 loss_all 和 test_acc 置零。
            loss_all = 0
            test_acc = 0.
            # 在每个 epoch 和 batch 开始时，设置随机种子确保可复现性
            deterministic(args.seed)
            for step, (data, target) in enumerate(test_data):
                deterministic(args.seed)
                # 将数据移动到指定设备
                data = data.to(device)
                # 如果 epoch 大于 -1，创建 input_u1 和 input_u2 的副本，并对 input_u1 添加噪声和旋转变换
                if epoch > -1:
                    
                    input_u1 = copy.deepcopy(data)
                    input_u2 = copy.deepcopy(data)
                    input_u1 = transforms_for_noise(input_u1, 0.5)
                    input_u1, rot_mask, flip_mask = transforms_for_rot(input_u1)
    
                
                # 将目标数据移动到指定设备
                target = target.to(device)
                # 使用模型对数据进行前向传播，计算输出 output
                output = model_adapt(data, "main")
                # 计算输出的熵损失 loss_entropy_before
                loss_entropy_before = entropy_loss(output, c=2)
                # 如果 epoch 大于 -1，计算变换后的输出 output_u1 和 output_u2，并将 output_u1 旋转回原始位置。计算一致性损失 consistency_loss
                if epoch > -1:
                    output_u1 = model_adapt(input_u1, "main")
                    output_u2 = model_adapt(input_u2, "main")
                    output_u1 = transforms_back_rot(output_u1, rot_mask, flip_mask)
                    consistency_loss = softmax_mse_loss(output_u1, output_u2)
                # 计算平滑损失 loss_smooth，并取其范数
                loss_smooth = smooth_loss(output)
                loss_smooth = torch.norm(loss_smooth)
                

                # 计算总损失 all_loss，如果 epoch 大于 -1，将熵损失、一致性损失和平滑损失加权相加
                all_loss = loss_entropy_before
                if epoch > -1:
                    all_loss = 10*all_loss + 0.1*torch.mean(consistency_loss) + loss_smooth 

                # 清零优化器的梯度
                optimizer.zero_grad()
                # 反向传播计算梯度
                all_loss.backward()
                # 使用优化器更新参数
                optimizer.step()
                # 使用更新后的模型对数据进行前向传播，计算输出 output
                output = model_adapt(data, "main")
                # 计算 Dice 损失 loss，并累计到 loss_all
                loss = criterion(output, target)
                loss_all += loss.item()
                # 计算 Dice 系数，并累计到 test_acc
                test_acc += DiceLoss().dice_coef(output, target).item()

            # 计算并打印每轮测试的准确率 report_acc
            report_acc = round(test_acc/len(test_data),4)
            print('Test Acc:', report_acc)
            # 如果当前轮的准确率大于 best_dice，更新 best_dice
            if best_dice < test_acc/len(test_data):
                best_dice = test_acc/len(test_data)
            # 将当前轮的准确率添加到 dice_buffer 并打印历史准确率
            dice_buffer.append(report_acc)
            print('Acc History:', dice_buffer)
            
        # 计算并打印最终的平均损失和准确率
        loss = loss_all / len(test_data)
        acc = test_acc/ len(test_data)
        print('Best Acc:', round(best_dice,4)) 
        # print(dice_buffer)
        # 将结果存储到 metrics 字典中，并返回该字典
        metrics['test_loss'] = loss
        metrics["test_dice"] = acc
        return metrics

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
                log_probs = model(x, "main")
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
                pred = model(x, "main")
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
