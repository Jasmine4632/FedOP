import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import logging
import copy
import torch.nn as nn

class ClientFedFed:
    def __init__(self, client_idx, local_training_data, local_val_data, local_test_data, local_sample_number, args, device, model_trainer):
        """
        初始化客户端对象
        """
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_val_data = local_val_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        logging.info(f"Client {client_idx} - Number of samples: {self.local_sample_number}")
        self.args = args
        self.device = device
        self.model_trainer = copy.deepcopy(model_trainer)
        self.model = copy.deepcopy(self.model_trainer.model)  # 每个客户端维护独立的模型副本

    def update_local_dataset(self, client_idx, local_training_data, local_val_data, local_test_data, local_sample_number):
        """
        更新客户端本地数据集
        """
        self.client_idx = client_idx
        self.model_trainer.set_id(client_idx)
        self.local_training_data = local_training_data
        self.local_val_data = local_val_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        """
        返回客户端的样本数量
        """
        return self.local_sample_number

    def train(self, w_global):
        """
        在本地数据集上训练模型
        """
        # 设置全局模型参数
        self.model_trainer.set_model_params(w_global)
        # 训练模型
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        # 获取训练后的模型参数
        weights = self.model_trainer.get_model_params()
        # 返回训练后的本地模型参数
        return weights

    def local_validate(self, local_param=None):
        """
        在本地验证集上评估模型
        """
        # 如果传入了特定参数，先设置该模型参数
        if local_param is not None:
            self.model_trainer.set_model_params(local_param)
        # 在验证数据集上测试模型性能
        metrics = self.model_trainer.test(self.local_val_data, self.device, self.args)
        return metrics

    def local_test(self, b_use_test_dataset=True, local_param=None):
        """
        在本地测试集或训练集上测试模型性能
        """
        # 如果传入了特定参数，先设置该模型参数
        if local_param is not None:
            self.model_trainer.set_model_params(local_param)

        # 选择测试数据集
        test_data = self.local_test_data if b_use_test_dataset else self.local_training_data
        # 如果测试数据集为空，返回默认值
        if test_data is None:
            logging.warning(f"Client {self.client_idx} has no test data.")
            return {'test_acc': 0.0, 'test_loss': 0.0}

        # 在测试数据集上测试模型性能
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics

    def ood_test(self, ood_data, w_global):
        """
        在OOD数据上评估模型性能
        """
        # 设置全局模型参数
        self.model_trainer.set_model_params(w_global)
        # 在OOD数据上测试模型性能
        metrics = self.model_trainer.test(ood_data, self.device, self.args)
        return metrics

    def test_time_adaptation_by_iopfl(self, w_global):
        """
        测试时间自适应（基于渐进联邦学习 - IoPFL）
        """
        if w_global is not None:
            self.model_trainer.set_model_params(w_global)
        metrics = self.model_trainer.io_pfl(self.local_test_data, self.device, self.args)
        return metrics
