import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import logging
import copy
import torch.nn as nn


class ClientProx:
    def __init__(self, client_idx, local_training_data, local_val_data, local_test_data, local_sample_number, args, device, model_trainer):
        """
        初始化客户端对象
        """
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_val_data = local_val_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        logging.info(f"Client {client_idx} - Number of samples: {local_sample_number}")
        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        self.model = copy.deepcopy(self.model_trainer.model)  # 确保每个客户端维护独立的模型副本

    def update_local_dataset(self, client_idx, local_training_data, local_val_data, local_test_data, local_sample_number):
        """
        更新客户端本地数据集
        """
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_val_data = local_val_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.model_trainer.set_id(client_idx)

    def get_sample_number(self):
        """
        返回本地样本数量
        """
        return self.local_sample_number

    def train(self, w_global):
        """
        本地训练方法（FedProx）
        """
        # 设置全局模型参数
        self.model_trainer.set_model_params(w_global)
        # 获取全局参数，用于计算 Proximal 项
        global_params = copy.deepcopy(list(self.model_trainer.model.parameters()))
        # 调用 FedProx 特有的训练方法
        self.model_trainer.train_prox(self.local_training_data, self.device, self.args, global_params)
        # 返回训练后的本地模型参数
        return self.model_trainer.get_model_params()

    def local_validate(self, local_param=None):
        """
        本地验证方法
        """
        if local_param is not None:
            self.model_trainer.set_model_params(local_param)
        return self.model_trainer.test(self.local_val_data, self.device, self.args)

    def local_test(self, b_use_test_dataset, local_param=None):
        """
        本地测试方法
        """
        if local_param is not None:
            self.model_trainer.set_model_params(local_param)
        data = self.local_test_data if b_use_test_dataset else self.local_training_data
        return self.model_trainer.test(data, self.device, self.args)

    def ood_test(self, ood_data, w_global):
        """
        OOD 测试方法
        """
        self.model_trainer.set_model_params(w_global)
        return self.model_trainer.test(ood_data, self.device, self.args)

    def test_time_adaptation_by_iopfl(self, w_global):
        """
        测试时间自适应方法
        """
        if w_global is not None:
            self.model_trainer.set_model_params(w_global)
        return self.model_trainer.io_pfl(self.local_test_data, self.device, self.args)
