import logging
import cv2
import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from .model_trainer import ModelTrainer
from utils.loss import DiceLoss, entropy_loss
from data.prostate.transforms import transforms_for_noise, transforms_for_rot, transforms_for_scale, transforms_back_scale, transforms_back_rot
from torch.cuda.amp import GradScaler, autocast
import copy
import numpy as np
import random
import traceback

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
                output = model_adapt(data)
                # 计算输出的熵损失 loss_entropy_before
                loss_entropy_before = entropy_loss(output, c=2)
                # 如果 epoch 大于 -1，计算变换后的输出 output_u1 和 output_u2，并将 output_u1 旋转回原始位置。计算一致性损失 consistency_loss
                if epoch > -1:
                    output_u1 = model_adapt(input_u1)
                    output_u2 = model_adapt(input_u2)
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
                output = model_adapt(data)
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
    
    def train_with_dynamic_weights(self, w_global, train_data, device, args, ps, p0, lamda, mu, model_cs, learning_rate, drlr, client_idx):
        # 设置全局模型参数
        self.set_model_params(w_global)
        self.model.to(device)
        self.model.train()

        criterion = DiceLoss().to(device)

        # 支持多种优化器
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr, amsgrad=True)

        # 初始化记录每轮损失和准确率的列表
        epoch_loss = []
        epoch_acc = []

        for epoch in range(args.wk_iters):
            batch_loss = []
            batch_acc = []

            for x, labels in train_data:
                optimizer.zero_grad()
                x, labels = x.to(device), labels.to(device)

                # 前向传播
                log_probs = self.model(x)
                loss = criterion(log_probs, labels)
                loss.backward()

                # 更新个性化模型参数
                for param_c, param in zip(model_cs[client_idx].parameters(), self.model.parameters()):
                    param_c.data -= learning_rate * param.grad * ps[client_idx]

                # 更新动态权重
                for cid in range(len(ps)):
                    cnt = 0
                    p_grad = 0
                    for param_c, param in zip(model_cs[cid].parameters(), self.model.parameters()):
                        p_grad += torch.mean(param.grad * param_c).item()
                        cnt += 1
                    p_grad = p_grad / cnt
                    p_grad += lamda * mu * (ps[cid] - p0[cid])
                    ps[cid] -= drlr * p_grad

                # 记录每个批次的损失和准确率
                acc = DiceLoss().dice_coef(log_probs, labels).item()
                batch_loss.append(loss.item())
                batch_acc.append(acc)

                # 更新模型参数
                optimizer.step()

            # 记录每轮的平均损失和准确率
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_acc.append(sum(batch_acc) / len(batch_acc))

            # 打印日志
            logging.info('Client Index = {}\tEpoch: {}\tAcc:{:.4f}\tLoss: {:.4f}'.format(
                client_idx, epoch, sum(epoch_acc) / len(epoch_acc), sum(epoch_loss) / len(epoch_loss)
            ))

        # 调整 lambda 值
        current_round = args.comm_round
        if current_round < args.L:
            lamda = (torch.cos(torch.tensor(current_round * torch.pi / args.L)) + 1) / 2
        else:
            lamda = 0

        # 返回更新后的模型参数
        return self.get_model_params()
    
    def train_prox(self, train_data, device, args, global_params):
        """
        联邦训练方法 (FedProx)
        :param train_data: 训练数据集
        :param device: 运行设备
        :param args: 参数配置
        :param global_params: 全局模型参数，用于计算 Proximal 项
        """
        model = self.model
        model.to(device)
        model.train()

        # 将 global_params 转移到设备上
        global_params = [param.to(device) for param in global_params]

        # 损失函数
        criterion = DiceLoss().to(device)

        # 优化器
        optimizer = (
            torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
            if args.client_optimizer == "sgd"
            else torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, amsgrad=True)
        )

        # 训练过程
        epoch_loss = []
        epoch_acc = []
        for epoch in range(args.wk_iters):
            batch_loss = []
            batch_acc = []
            for x, labels in train_data:
                model.zero_grad()
                x, labels = x.to(device), labels.to(device)

                # 前向传播
                log_probs = model(x)
                loss = criterion(log_probs, labels)

                # 添加 FedProx 的正则项
                prox_term = 0
                for param, global_param in zip(model.parameters(), global_params):
                    prox_term += ((param - global_param) ** 2).sum()
                loss += (args.mu / 2) * prox_term  # 加入 Proximal 项

                # 反向传播
                loss.backward()
                optimizer.step()

                # 记录准确率和损失
                acc = DiceLoss().dice_coef(log_probs, labels).item()
                batch_loss.append(loss.item())
                batch_acc.append(acc)

            # 记录每个 epoch 的平均损失和准确率
            avg_epoch_loss = sum(batch_loss) / len(batch_loss)
            avg_epoch_acc = sum(batch_acc) / len(batch_acc)
            epoch_loss.append(avg_epoch_loss)
            epoch_acc.append(avg_epoch_acc)

            # 日志记录
            logging.info(
                f"FedProx - Client Index = {getattr(self, 'id', 'unknown')}\t"
                f"Epoch: {epoch}\tAcc: {avg_epoch_acc:.4f}\tLoss: {avg_epoch_loss:.4f}"
            )

        # 返回训练日志
        return {"epoch_loss": epoch_loss, "epoch_acc": epoch_acc}

    def traindg(self, train_data, device, args, ood_data=None):
        """
        使用元学习内循环和外循环在本地数据上进行训练
        训练过程包括内外循环以及模型参数更新
        如果提供了 OOD 数据，进行领域泛化训练
        """
        model = self.model
        model.to(device)
        model.train()

        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = (
            torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
            if args.client_optimizer == "sgd"
            else torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, amsgrad=True)
        )

        epoch_loss = []
        epoch_acc = []

        # **1. 执行内循环训练 (Meta-learning inner loop)**
        inner_model_params = self.inner_loop_train(train_data, model, criterion, optimizer, device, args, epoch_loss, epoch_acc)

        # **2. 使用 OOD 数据进行领域泛化训练 (Domain Generalization)**
        # if ood_data is not None:
        #     self.domain_generalization(ood_data, model, criterion, optimizer, device, args)

        # **3. 执行外循环训练 (Meta-learning outer loop)**
        # 在内循环更新后的模型参数上进行外循环训练，以更新全局模型
        local_model_params = self.outer_loop_train(train_data, inner_model_params, model, criterion, optimizer, device, args, epoch_loss, epoch_acc)

        # 输出最后一个 epoch 的损失和准确率
        logging.info('Final Epoch - Acc:{:.4f}\tLoss: {:.4f}'.format(epoch_acc[-1], epoch_loss[-1]))

        # 返回训练后的模型参数
        return local_model_params

    def inner_loop_train(self, train_data, model, criterion, optimizer, device, args, epoch_loss, epoch_acc):
        """
        执行元学习的内循环训练
        在本地数据上进行快速训练，目的是让模型适应本地数据
        """
        model.train()
        for epoch in range(args.inner_lr_steps):  # 迭代内循环的步数
            batch_loss = []
            batch_acc = []
            for batch_idx, (inputs, targets) in enumerate(train_data):
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.squeeze(1)
                targets = targets.long() 
                optimizer.zero_grad()

                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                acc = (outputs.argmax(dim=1) == targets).float().mean().item()

                # 反向传播
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
                batch_acc.append(acc)

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_acc.append(sum(batch_acc) / len(batch_acc))

        return model.state_dict()

    def outer_loop_train(self, train_data, inner_model_params, model, criterion, optimizer, device, args, epoch_loss, epoch_acc):
        """
        执行元学习的外循环训练
        在经过内循环训练后，对全局模型进行优化
        """
        # 加载内循环更新后的模型参数
        model.load_state_dict(inner_model_params)
        model.train()

        for epoch in range(args.outer_lr_steps):  # 外循环训练步数
            batch_loss = []
            batch_acc = []
            for batch_idx, (inputs, targets) in enumerate(train_data):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()

                # 前向传播
                outputs = model(inputs)
                targets = targets.squeeze(1)  

                targets = targets.long()  
                loss = criterion(outputs, targets)

                acc = (outputs.argmax(dim=1) == targets).float().mean().item()

                # 反向传播
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
                batch_acc.append(acc)

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_acc.append(sum(batch_acc) / len(batch_acc))

        return model.state_dict()


    def domain_generalization(self, ood_data, model, criterion, optimizer, device, args):
        """
        在 OOD 数据上进行训练，提升模型的领域泛化能力
        """
        model.train()
        for epoch in range(args.ood_train_steps):  # OOD 训练步数
            for batch_idx, (inputs, targets) in enumerate(ood_data):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()

                # 前向传播
                outputs = model(inputs)
                targets = targets.squeeze(1)  

                targets = targets.long()  
                loss = criterion(outputs, targets)

                # 反向传播
                loss.backward()
                optimizer.step()

        print("OOD domain generalization complete.")

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

class ModelTrainerSegmentationPer(ModelTrainer):
    """
    个性化联邦学习的模型训练器（PerFedAvg）
    """
    def __init__(self, model, args):
        """
        初始化：继承自 ModelTrainer，同时初始化个性化模型
        """
        super().__init__(model, args)  # 调用父类构造方法
        self.personal_model = copy.deepcopy(self.model)  # 初始化个性化模型

    def get_model_params(self, model_type="global"):
        """
        获取模型参数
        :param model_type: "global" 或 "personal"
        """
        if model_type == "global":
            return self.model.cpu().state_dict()
        elif model_type == "personal":
            return self.personal_model.cpu().state_dict()
        else:
            raise ValueError("Invalid model type. Use 'global' or 'personal'.")

    def set_model_params(self, model_parameters, model_type="global"):
        """
        设置模型参数
        :param model_type: "global" 或 "personal"
        """
        if model_type == "global":
            self.model.load_state_dict(model_parameters)
        elif model_type == "personal":
            self.personal_model.load_state_dict(model_parameters)
        else:
            raise ValueError("Invalid model type. Use 'global' or 'personal'.")

    def train(self, train_data, device, args, model_type="global"):
        """
        训练模型
        :param train_data: 本地训练数据
        :param device: 训练设备
        :param args: 参数配置
        :param model_type: "global" 或 "personal"
        """
        if model_type == "global":
            model = self.model
        elif model_type == "personal":
            model = self.personal_model
        else:
            raise ValueError("Invalid model type. Use 'global' or 'personal'.")

        model.to(device)
        model.train()

        # 定义损失函数
        criterion = DiceLoss().to(device)

        # 定义优化器
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr,
                amsgrad=True,
            )

        # 开始训练
        epoch_loss = []
        epoch_acc = []
        for epoch in range(args.wk_iters):
            batch_loss = []
            batch_acc = []
            for x, labels in train_data:
                x, labels = x.to(device), labels.to(device)

                # 前向传播
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, labels)
                acc = DiceLoss().dice_coef(outputs, labels).item()

                # 反向传播
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
                batch_acc.append(acc)

            # 记录每个 epoch 的损失和准确率
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_acc.append(sum(batch_acc) / len(batch_acc))

            logging.info(
                f"Client Index = {self.id}\tModel Type = {model_type}\tEpoch: {epoch}\tAcc: {sum(epoch_acc) / len(epoch_acc):.4f}\tLoss: {sum(epoch_loss) / len(epoch_loss):.4f}"
            )

    def test(self, test_data, device, args, model_type="global"):
        """
        测试模型
        :param test_data: 测试数据
        :param device: 测试设备
        :param args: 参数配置
        :param model_type: "global" 或 "personal"
        """
        if model_type == "global":
            model = copy.deepcopy(self.model)
        elif model_type == "personal":
            model = copy.deepcopy(self.personal_model)
        else:
            raise ValueError("Invalid model type. Use 'global' or 'personal'.")

        model.to(device)
        model.eval()

        metrics = {"test_loss": 0, "test_acc": 0}
        criterion = DiceLoss().to(device)

        with torch.no_grad():
            for x, labels in test_data:
                x, labels = x.to(device), labels.to(device)

                # 前向传播
                outputs = model(x)
                loss = criterion(outputs, labels)
                acc = DiceLoss().dice_coef(outputs, labels).item()

                metrics["test_loss"] += loss.item() * labels.size(0)
                metrics["test_acc"] += acc

        # 平均损失和准确率
        metrics["test_loss"] /= len(test_data.dataset)
        metrics["test_acc"] /= len(test_data)

        return metrics

class ModelTrainerSegmentationDitto(ModelTrainer): 
    def __init__(self, model, args):
        super().__init__(model, args)
        self.personal_model = copy.deepcopy(model)  # 初始化个性化模型

    def get_model_params(self, model_type="global"):
        """
        获取模型参数
        :param model_type: 模型类型，支持 'global' 和 'personal'
        """
        if model_type == "global":
            return self.model.cpu().state_dict()
        elif model_type == "personal":
            return self.personal_model.cpu().state_dict()
        else:
            raise ValueError("Invalid model type. Use 'global' or 'personal'.")

    def set_model_params(self, model_parameters, model_type="global"):
        """
        设置模型参数
        :param model_parameters: 要加载的模型参数
        :param model_type: 模型类型，支持 'global' 和 'personal'
        """
        if model_type == "global":
            self.model.load_state_dict(model_parameters)
        elif model_type == "personal":
            self.personal_model.load_state_dict(model_parameters)
        else:
            raise ValueError("Invalid model type. Use 'global' or 'personal'.")

    # 训练方法，支持全局和个性化模型
    def train(self, train_data, device, args, model_type="global"):
        """
        模型训练方法
        :param train_data: 训练数据集
        :param device: 设备（CPU 或 GPU）
        :param args: 超参数设置
        :param model_type: 模型类型，支持 'global' 和 'personal'
        """
        if model_type == "global":
            model = self.model
        elif model_type == "personal":
            model = self.personal_model
        else:
            raise ValueError("Invalid model type. Use 'global' or 'personal'.")

        model.to(device)
        model.train()

        # 损失函数
        criterion = DiceLoss().to(device)

        # 优化器
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, amsgrad=True)

        epoch_loss = []
        epoch_acc = []

        for epoch in range(args.wk_iters):
            batch_loss = []
            batch_acc = []

            for batch_idx, (x, labels) in enumerate(train_data):
                optimizer.zero_grad()
                x, labels = x.to(device), labels.to(device)
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                acc = DiceLoss().dice_coef(log_probs, labels).item()
                loss.backward()
                optimizer.step()
                
                # # 打印中间值调试
                # print(f"Log probs: {log_probs.shape}, Labels: {labels.shape}")
                # print(f"Loss: {loss.item()}, Accuracy: {acc}")
                
                batch_loss.append(loss.item())
                batch_acc.append(acc)
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_acc.append(sum(batch_acc) / len(batch_acc))
            logging.info('Client Index = {}\tModel Type: {}\tEpoch: {}\tAcc:{:.4f}\tLoss: {:.4f}'.format(
                self.id, model_type, epoch, sum(epoch_acc) / len(epoch_acc), sum(epoch_loss) / len(epoch_loss)))

    # 测试方法，支持全局和个性化模型
    def test(self, test_data, device, args, ood=False, model_type="global"):
        """
        模型测试方法
        :param test_data: 测试数据集
        :param device: 设备（CPU 或 GPU）
        :param args: 超参数设置
        :param ood: 是否进行分布外测试
        :param model_type: 模型类型，支持 'global' 和 'personal'
        """
        if model_type == "global":
            model = copy.deepcopy(self.model)
        elif model_type == "personal":
            model = copy.deepcopy(self.personal_model)
        else:
            raise ValueError("Invalid model type. Use 'global' or 'personal'.")

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
