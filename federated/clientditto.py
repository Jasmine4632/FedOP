import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import logging
import copy
import torch
import torch.nn as nn

class ClientDitto:
    # 初始化方法
    def __init__(self, client_idx, local_training_data, local_val_data, local_test_data, local_sample_number, args, device, model_trainer):
        """
        初始化客户端对象。
        """
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_val_data = local_val_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.args = args
        self.device = device

        # 深度复制模型训练器，确保每个客户端有独立的模型副本
        self.model_trainer = copy.deepcopy(model_trainer)
        self.model = copy.deepcopy(self.model_trainer.model)  # 客户端模型
        self.personal_model = copy.deepcopy(self.model_trainer.personal_model)  # 个性化模型
        self.mu = args.mu

        logging.info(f"Client {client_idx}: local_sample_number = {self.local_sample_number}")

    # 更新本地数据集
    def update_local_dataset(self, client_idx, local_training_data, local_val_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_val_data = local_val_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    # 获取样本数量
    def get_sample_number(self):
        return self.local_sample_number

    # 训练方法：包含全局模型训练和个性化模型训练
    def train_with_personalization(self, w_global):
        """
        使用全局模型参数训练，并进行个性化调整。
        """
        # 1. 设置全局模型参数
        self.model_trainer.set_model_params(w_global, model_type="global")

        # 2. 在本地数据上训练全局模型
        self.model_trainer.train(self.local_training_data, self.device, self.args, model_type="global")

        # 3. 获取全局模型训练后的参数
        updated_global_params = self.model_trainer.get_model_params(model_type="global")

        # 4. 在本地数据上训练个性化模型
        self.model_trainer.train(self.local_training_data, self.device, self.args, model_type="personal")

        # 5. 正则化约束：调整个性化模型与全局模型的权重差异
        with torch.no_grad():
            personal_params = self.model_trainer.get_model_params(model_type="personal")
            for key in personal_params.keys():
                # 确保张量是浮点数类型
                personal_params[key] = personal_params[key].float()
                w_global[key] = w_global[key].float()

                # 应用正则化
                personal_params[key] -= self.mu * (personal_params[key] - w_global[key])

        # 6. 更新个性化模型参数
        self.model_trainer.set_model_params(personal_params, model_type="personal")

        # 7. 返回更新后的全局模型参数（用于服务器端聚合）
        return updated_global_params

    # 本地验证方法
    def local_validate(self, model_type="personal"):
        """
        在本地验证数据上测试模型。
        """
        return self.model_trainer.test(self.local_val_data, self.device, self.args, model_type=model_type)

    # 本地测试方法
    def local_test(self, b_use_test_dataset=True, model_type="personal"):
        """
        在本地测试数据上测试模型。
        """
        test_data = self.local_test_data if b_use_test_dataset else self.local_training_data
        return self.model_trainer.test(test_data, self.device, self.args, model_type=model_type)

    def ood_test(self, ood_data, w_global):
        self.model_trainer.set_model_params(w_global)
        metrics = self.model_trainer.test(ood_data, self.device, self.args)
        return metrics