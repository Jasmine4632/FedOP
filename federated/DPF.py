from typing import Optional
import logging
import numpy as np
import torch
import torch.nn as nn
import copy
import random
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import random
import numpy as np

import torch
import torch.nn.functional as F


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
    
class ALA:
    def __init__(self,
                 train_data: DataLoader,
                 rand_percent: int,
                 ala_times: int,
                 client_idx,
                 device,
                 layer_idx: int,
                 eta: float = 1.0,
                 alpha: float = 0.5,   # 新增本地梯度影响因子
                 lambda_reg: float = 0.1,  # 新增正则化系数
                 threshold: float = 0.1,
                 num_pre_loss: int = 10
                ) -> None:
        self.device = device 
        self.client_idx = client_idx  # 新增的属性
        self.loss_function = DiceLoss()  # 或者使用 nn.MSELoss() 根据任务而定
        self.train_data = train_data
        self.ala_times = ala_times
        self.rand_percent = rand_percent
        self.layer_idx = int(layer_idx)
        self.eta = eta
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.threshold = threshold
        self.num_pre_loss = num_pre_loss
        self.device = device
        
        self.weights = None
        self.start_phase = True
        self.ala_round = 0  # 记录执行的轮次
        
    def adaptive_local_aggregation(self, global_model: nn.Module, local_model: nn.Module, task="normal") -> None:
        logging.info(f"Client {self.client_idx}: Starting adaptive_local_aggregation")

        # Ensure models are on the correct device
        global_model = global_model.to(self.device)
        local_model = local_model.to(self.device)

        # Randomly sample partial local training data
        dataset = self.train_data.dataset  # 获取原始的 dataset
        rand_ratio = self.rand_percent / 100
        rand_num = int(rand_ratio * len(dataset))  # 计算随机采样的数量

        # Randomly sample partial local training data
        dataset = self.train_data.dataset  # 获取原始的 dataset
        rand_ratio = self.rand_percent / 100
        rand_num = int(rand_ratio * len(dataset))  # 计算随机采样的数量

        if rand_num > 0:
            # 确保随机索引在数据集范围内
            rand_idx = random.sample(range(len(dataset)), rand_num)  # 生成不重复的随机索引
            subset_dataset = Subset(dataset, rand_idx)  # 创建子集
        else:
            subset_dataset = dataset  # 如果 rand_num 为 0，保留原始数据集

        # 使用新的 batch_size 重新创建 DataLoader
        rand_loader = DataLoader(subset_dataset, batch_size=self.ala_times, shuffle=True, drop_last=False)
          
        # Obtain the references of the parameters
        params_g = list(global_model.parameters())
        params = list(local_model.parameters())
        
        # Check if parameters of global and local model are identical
        identical_count = 0
        for param_g, param in zip(global_model.parameters(), local_model.parameters()):
            if torch.equal(param_g, param):
                identical_count += 1

        logging.info(f"Number of identical parameters between global and local model: {identical_count}/{len(params_g)}")
        
        # Deactivate ALA at the 1st communication iteration if models are identical
        if torch.sum(params_g[0] - params[0]) == 0:
            return

        # Preserve all the updates in the lower layers
        for param, param_g in zip(params[:-self.layer_idx], params_g[:-self.layer_idx]):
            param.data = param_g.data.clone()

        # Temp local model only for weight learning
        model_t = copy.deepcopy(local_model).to(self.device)
        params_t = list(model_t.parameters())

        # Only consider higher layers
        params_p = params[-self.layer_idx:]
        params_gp = params_g[-self.layer_idx:]
        params_tp = params_t[-self.layer_idx:]

        # 冻结前 len(params_t) - self.layer_idx 个参数
        for param in params_t[:-self.layer_idx]:
            param.requires_grad = False

        # Reinitialize optimizer to ensure it includes all trainable parameters
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_t.parameters()), lr=0, amsgrad=True)
        
        # Initialize the weight to all ones in the beginning
        if self.weights is None:
            self.weights = [torch.ones_like(param.data).to(self.device) for param in params_p]

        # initialize the higher layers in the temp local model
        for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp, self.weights):
            param_t.data = param + weight.sigmoid() * (param_g - param)

        # Weight learning
        losses = []  # Record losses
        cnt = 0  # Weight training iteration counter
        while True:
            for batch_idx, (x, labels) in enumerate(rand_loader):
                optimizer.zero_grad()
                x, labels = x.to(self.device), labels.to(self.device)

                log_probs = model_t(x)
                # print(f"Final model output shape: {log_probs.shape}")
                criterion = DiceLoss().to(self.device)  # 首先创建 DiceLoss 的实例
                loss = criterion(log_probs, labels)  # 使用 criterion(log_probs, labels) 来计算损失
                loss.backward()  # 对损失进行反向传播

                for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp, self.weights):
                    grad = param_t.grad
                    norm_diff = torch.norm(param_g - param) + 1e-8
                    weight.data -= self.eta * (grad * (param_g - param) / norm_diff + self.lambda_reg * weight)

                for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp, self.weights):
                    param_t.data = param + weight.sigmoid() * (param_g - param) - self.alpha * param_t.grad
  

            losses.append(loss.item())
            cnt += 1

            # only train one epoch in the subsequent iterations
            if not self.start_phase:
                break

            # train the weight until convergence
            if len(losses) > self.num_pre_loss and np.std(losses[-self.num_pre_loss:]) < self.threshold:
                print('Client:',  np.std(losses[-self.num_pre_loss:]),
                    '\tALA epochs:', cnt)
                break
                       
        self.start_phase = False

        # obtain initialized local model
        for param, param_t in zip(params_p, params_tp):
            param.data = param_t.data.clone()
