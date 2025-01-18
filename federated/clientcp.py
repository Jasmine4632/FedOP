import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import logging
import copy
import torch.nn as nn

class ClientCP:
    def __init__(self, client_idx, local_training_data, local_val_data, local_test_data, local_sample_number, args, device, model_trainer):
        """
        初始化客户端
        """
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_val_data = local_val_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        self.model = copy.deepcopy(self.model_trainer.model)
        logging.info(f"Client {client_idx} initialized with {self.local_sample_number} samples.")

    def update_local_dataset(self, client_idx, local_training_data, local_val_data, local_test_data, local_sample_number):
        """
        更新客户端本地数据集
        """
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_val_data = local_val_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        """
        获取本地样本数量
        """
        return self.local_sample_number

    def train(self, w_global):
        """
        本地训练：设置全局模型参数，训练本地模型，返回本地更新后的参数
        """
        # 设置全局模型参数
        self.model_trainer.set_model_params(w_global)

        # 使用本地数据训练
        self.model_trainer.train(self.local_training_data, self.device, self.args)

        # 获取更新后的本地模型参数
        updated_weights = self.model_trainer.get_model_params()
        return updated_weights

    def local_validate(self):
        """
        本地验证：返回验证集上的性能指标
        """
        metrics = self.model_trainer.test(self.local_val_data, self.device, self.args)
        return metrics

    def local_test(self, use_test_dataset=True):
        """
        本地测试：在测试集或训练集上测试模型
        """
        if use_test_dataset:
            data = self.local_test_data
        else:
            data = self.local_training_data

        metrics = self.model_trainer.test(data, self.device, self.args)
        return metrics

    def ood_test(self, ood_data, w_global):
        """
        分布外 (OOD) 测试
        """
        self.model_trainer.set_model_params(w_global)
        metrics = self.model_trainer.test(ood_data, self.device, self.args)
        logging.info(f"Client {self.client_idx} OOD Test - Acc: {metrics['test_acc']:.4f}, Loss: {metrics['test_loss']:.4f}")
        return metrics

    def test_time_adaptation(self, w_global):
        """
        测试时间自适应
        """
        if w_global is not None:
            self.model_trainer.set_model_params(w_global)
        metrics = self.model_trainer.io_pfl(self.local_test_data, self.device, self.args)
        logging.info(f"Client {self.client_idx} Test Time Adaptation - Acc: {metrics['test_acc']:.4f}, Loss: {metrics['test_loss']:.4f}")
        return metrics
