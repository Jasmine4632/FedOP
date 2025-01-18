import copy
import logging
import torch

class ClientPer:
    def __init__(self, client_idx, local_training_data, local_val_data, local_test_data, local_sample_number, args, device, model_trainer):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_val_data = local_val_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        logging.info(f"Client {client_idx}: self.local_sample_number = {self.local_sample_number}")
        self.args = args
        self.device = device
        self.model_trainer = model_trainer

    def update_local_dataset(self, client_idx, local_training_data, local_val_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.model_trainer.set_id(client_idx)
        self.local_training_data = local_training_data
        self.local_val_data = local_val_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number

    def train_with_personalization(self, w_global):
        """
        使用全局模型参数进行个性化训练
        """
        # 1. 设置全局模型参数
        self.model_trainer.set_model_params(w_global, model_type="global")

        # 2. 在本地数据上训练全局模型
        self.model_trainer.train(self.local_training_data, self.device, self.args, model_type="global")

        # 3. 获取训练后的全局模型参数
        updated_global_params = self.model_trainer.get_model_params(model_type="global")

        # 4. 个性化训练
        self.model_trainer.set_model_params(updated_global_params, model_type="personal")
        personal_params = self.model_trainer.get_model_params(model_type="personal")

        # 5. 使用正则化调整个性化模型，使其接近全局模型
        with torch.no_grad():
            for key in personal_params.keys():
                # 确保张量为浮点类型
                personal_params[key] = personal_params[key].float()
                w_global[key] = w_global[key].float()
                personal_params[key] -= self.args.mu * (personal_params[key] - w_global[key])

        # 6. 更新个性化模型参数
        self.model_trainer.set_model_params(personal_params, model_type="personal")

        # 返回更新后的全局模型参数
        return personal_params

    def local_validate(self, model_type="personal"):
        """
        使用个性化模型或全局模型进行验证
        """
        metrics = self.model_trainer.test(self.local_val_data, self.device, self.args, model_type=model_type)
        return metrics

    def local_test(self, b_use_test_dataset=True, model_type="personal"):
        """
        使用个性化模型或全局模型进行测试
        """
        test_data = self.local_test_data if b_use_test_dataset else self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args, model_type=model_type)
        return metrics

    def ood_test(self, ood_data, w_global):
        self.model_trainer.set_model_params(w_global)
        metrics = self.model_trainer.test(ood_data, self.device, self.args)
        return metrics