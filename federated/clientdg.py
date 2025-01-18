import copy
import logging
import torch
from torch import nn, optim

class ClientDG:
    def __init__(self, client_idx, local_training_data, local_val_data, local_test_data, local_sample_number, ood_data, args, device, model_trainer):
        """
        client_idx: 客户端索引
        local_training_data: 本地训练数据
        local_val_data: 本地验证数据
        local_test_data: 本地测试数据
        local_sample_number: 本地训练数据样本数量
        ood_data: 用于领域泛化的 OOD 数据
        args: 配置参数
        device: 训练设备
        model_trainer: 模型训练器
        """
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_val_data = local_val_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        logging.info(f"Client {self.client_idx} initialized with {self.local_sample_number} samples.")
        self.ood_data = ood_data
        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        self.model = copy.deepcopy(self.model_trainer.model)  # 确保每个客户端的模型是独立的副本

    def update_local_dataset(self, client_idx, local_training_data, local_val_data, local_test_data, local_sample_number, ood_data):
        """更新本地数据集及 OOD 数据"""
        self.client_idx = client_idx
        self.model_trainer.set_id(client_idx)  # 确保 model_trainer 有 set_id 方法
        self.local_training_data = local_training_data
        self.local_val_data = local_val_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.ood_data = ood_data

    def get_sample_number(self):
        """返回本地样本数量"""
        return self.local_sample_number

    def train(self, w_global):
        """本地训练过程：使用 model_trainer 调用训练方法"""
        # 设置全局模型参数
        self.model_trainer.set_model_params(w_global)

        local_model_params = self.model_trainer.traindg(self.local_training_data, self.device, self.args)

        return local_model_params

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

    def ood_test(self, ood_data, w_global):
        self.model_trainer.set_model_params(w_global)
        metrics = self.model_trainer.test(ood_data, self.device, self.args)
        return metrics

    def test_time_adaptation_by_iopfl(self, w_global=None):
        """测试时间自适应"""
        logging.info(f"Client {self.client_idx} starting test time adaptation.")
        if w_global is not None:
            self.model_trainer.set_model_params(w_global)
            logging.info(f"Client {self.client_idx} set global model parameters for adaptation.")
        metrics = self.model_trainer.io_pfl(self.local_test_data, self.device, self.args)
        logging.info(f"Client {self.client_idx} test time adaptation results: {metrics}")
        return metrics
