import logging
import torch
import os
import copy


class ClientIOP:
    # 初始化方法 __init__：
    # 初始化客户端对象，设置客户端索引、本地训练/验证/测试数据、样本数量、参数、设备和模型训练器。
    # 初始化 trajectory 和 prev_weight 为 None。
    def __init__(self, client_idx, local_training_data, local_val_data, local_test_data, local_sample_number, args,
                 device,
                 model_trainer):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_val_data = local_val_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        logging.info("self.local_sample_number = " + str(self.local_sample_number))

        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        self.trajectory = None
        self.prev_weight = None

    # 更新本地数据集方法 update_local_dataset：
    # 更新客户端的数据集及相关信息。
    def update_local_dataset(self, client_idx, local_training_data, local_val_data, local_test_data,
                             local_sample_number):
        self.client_idx = client_idx
        self.model_trainer.set_id(client_idx)
        self.local_training_data = local_training_data
        self.local_val_data = local_val_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    # 获取样本数量方法 get_sample_number：
    # 返回本地样本数量
    def get_sample_number(self):
        return self.local_sample_number

    # 训练方法 train：
    # 设置全局模型参数。
    # 在本地数据上训练模型。
    # 计算并更新模型参数轨迹。
    # 返回本地训练后的模型参数。
    def train(self, w_global):
        # 设置全局模型参数
        self.model_trainer.set_model_params(w_global)
        # 这里的model_trainer是指ModelTrainerSegmentation(ModelTrainer)，这句话是在调用ModelTrainerSegmentation的train
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        # 获取训练后的模型参数
        weights = self.model_trainer.get_model_params()
        # 计算并更新模型参数轨迹
        self.calcuate_trajectory(weights)
        # 保存当前模型参数作为前一次的权重
        self.prev_weight = weights
        # 返回本地训练后的模型参数
        return weights

    # 本地验证方法 local_validate：
    # 设置模型参数（如果提供）。
    # 在本地验证数据上测试模型，返回验证指标。
    def local_validate(self, local_param=None):
        if local_param is not None:
            self.model_trainer.set_model_params(local_param)
        metrics = self.model_trainer.test(self.local_val_data, self.device, self.args)
        return metrics

    # 本地测试方法 local_test：
    # 设置模型参数（如果提供）。
    # 根据参数选择使用测试数据集或训练数据集进行测试，返回测试指标。
    def local_test(self, b_use_test_dataset, local_param=None):
        if local_param is not None:
            self.model_trainer.set_model_params(local_param)

        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics

    # OOD 测试方法 ood_test：
    # 设置全局模型参数。
    # 在 OOD 数据上测试模型，返回测试指标
    def ood_test(self, ood_data, w_global):
        self.model_trainer.set_model_params(w_global)
        metrics = self.model_trainer.test(ood_data, self.device, self.args)
        return metrics

    # 按轨迹本地测试方法 local_test_by_trajectory：
    # 使用模型参数轨迹设置模型。
    # 在本地测试数据上测试模型，返回测试指标
    def local_test_by_trajectory(self):
        model_trainer_copy = copy.deepcopy(self.model_trainer)
        model_trainer_copy.set_model_params(self.trajectory)
        metrics = model_trainer_copy.test(self.local_test_data, self.device, self.args)
        del model_trainer_copy
        return metrics

    # 按轨迹本地验证方法 local_validate_by_trajectory：
    # 使用模型参数轨迹设置模型。
    # 在本地验证数据上测试模型，返回验证指标。
    def local_validate_by_trajectory(self):
        model_trainer_copy = copy.deepcopy(self.model_trainer)
        model_trainer_copy.set_model_params(self.trajectory)
        metrics = model_trainer_copy.test(self.local_val_data, self.device, self.args)
        del model_trainer_copy
        return metrics

    # 按轨迹 OOD 测试方法 ood_test_by_trajectory：
    # 使用模型参数轨迹设置模型。
    # 在 OOD 数据上测试模型，返回测试指标。
    def ood_test_by_trajectory(self, ood_test_data):
        model_trainer_copy = copy.deepcopy(self.model_trainer)
        model_trainer_copy.set_model_params(self.trajectory)
        metrics = model_trainer_copy.test(ood_test_data, self.device, self.args)
        del model_trainer_copy
        return metrics

    # 保存轨迹方法 save_trajectory：
    # 保存模型参数轨迹到指定路径。
    def save_trajectory(self, comm_round):
        torch.save(self.trajectory, os.path.join(self.args.save_path,"{}_idx_{}_round{}".format(self.args.mode, self.client_idx,comm_round)))

    # 计算轨迹方法 calcuate_trajectory：
    # 计算并更新模型参数轨迹
    def calcuate_trajectory(self, w_local):
        if self.trajectory == None:
            self.trajectory = w_local  # 轨迹也是另外一种模型参数  目的是使用这些历史参数的加权平均值来进行测试，而不是使用某一轮的具体模型参数。这样做的好处是可以减轻模型参数由于数据异质性（Non-IID）或者噪声造成的波动，从而获得更平滑和稳健的模型表现。
        else:
            for k in w_local.keys():
                self.trajectory[k] = self.args.alpha * self.trajectory[k] + (1 - self.args.alpha) * w_local[k]

    # 测试时间自适应方法 test_time_adaptation_by_iopfl：
    # 设置全局模型参数（如果提供）。
    # 调用模型训练器的 io_pfl 方法进行测试时间自适应的渐进联邦学习（PFL），返回测试指标。
    def test_time_adaptation_by_iopfl(self, w_global):
        # 如果传入了全局模型参数 w_global，则设置模型参数为该全局模型参数。这确保模型从一个经过训练的状态开始，而不是随机初始化的状态
        if w_global != None:
            self.model_trainer.set_model_params(w_global)
        metrics = self.model_trainer.io_pfl(self.local_test_data, self.device, self.args)
        return metrics
