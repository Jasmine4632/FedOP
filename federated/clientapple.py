import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import logging
import copy
import torch.nn as nn


class ClientAPPLE:
    def __init__(self, client_idx, local_training_data, local_val_data, local_test_data, local_sample_number, args, device, model_trainer, train_data_local_dict=None):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_val_data = local_val_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        self.model = copy.deepcopy(self.model_trainer.model)  # 独立模型副本

        # 初始化模型副本集合（model_cs）
        self.model_cs = [copy.deepcopy(self.model) for _ in range(len(train_data_local_dict))] if train_data_local_dict else []

        # 动态推断 num_clients
        self.num_clients = len(train_data_local_dict) if train_data_local_dict is not None else 1
        self.ps = [1 / self.num_clients for _ in range(self.num_clients)]  # 动态权重
        self.p0 = [1 / self.num_clients for _ in range(self.num_clients)]  # 初始化为均匀分布
        self.lamda = 1
        self.mu = args.mu
        self.L = int(args.L * args.comm_round)
        self.learning_rate = args.lr * self.num_clients
        self.drlr = args.dr_learning_rate
        
    def get_model_params(self):
        return self.model_trainer.get_model_params()  # Delegating to model_trainer's get_model_params method
    
    def train(self, w_global):
        # 调用模型训练方法，传递完整上下文
        return self.model_trainer.train_with_dynamic_weights(
            w_global,
            self.local_training_data,
            self.device,
            self.args,
            self.ps,
            self.p0,
            self.lamda,
            self.mu,
            self.model_cs,
            self.learning_rate,
            self.drlr,
            self.client_idx  # 传递客户端索引
        )

    # 更新本地数据集方法 update_local_dataset：
    # 更新客户端的数据集及相关信息。
    def set_dynamic_weights(self, global_weights):
        """
        设置客户端的动态权重。
        :param global_weights: 全局动态权重列表。
        """
        self.ps = global_weights
        
    def update_local_dataset(self, client_idx, local_training_data, local_val_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.model_trainer.set_id(client_idx)
        self.local_training_data = local_training_data
        self.local_val_data = local_val_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    # 获取样本数量方法 get_sample_number：
    # 返回本地样本数量。
    def get_sample_number(self):
        return self.local_sample_number

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
        test_data = self.local_test_data if b_use_test_dataset else self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics

    # OOD 测试方法 ood_test：
    # 设置全局模型参数。
    # 在 OOD 数据上测试模型，返回测试指标。
    def ood_test(self, ood_data, w_global):
        self.model_trainer.set_model_params(w_global)
        metrics = self.model_trainer.test(ood_data, self.device, self.args)
        return metrics

    # 测试时间自适应方法 test_time_adaptation_by_iopfl：
    # 设置全局模型参数（如果提供）。
    # 调用模型训练器的 io_pfl 方法进行测试时间自适应的渐进联邦学习（PFL），返回测试指标。
    def test_time_adaptation_by_iopfl(self, w_global):
        if w_global is not None:
            self.model_trainer.set_model_params(w_global)
        metrics = self.model_trainer.io_pfl(self.local_test_data, self.device, self.args)
        return metrics
