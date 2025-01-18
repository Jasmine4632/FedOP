import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import logging
import copy
import torch.nn as nn

class ClientAvg:
    # 初始化方法 __init__：
    # 初始化客户端对象，设置客户端索引、本地训练/验证/测试数据、样本数量、参数、设备和模型训练器。
    # 初始化 trajectory 和 prev_weight 为 None。
    def __init__(self, client_idx, local_training_data, local_val_data, local_test_data, local_sample_number, args, device,
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
        self.model = copy.deepcopy(self.model_trainer.model)  # 这里用 deepcopy 保证了每个客户端的模型是独立的副本


    # 更新本地数据集方法 update_local_dataset：
    # 更新客户端的数据集及相关信息。
    def update_local_dataset(self, client_idx, local_training_data, local_val_data, local_test_data, local_sample_number):
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


    # 测试时间自适应方法 test_time_adaptation_by_iopfl：
    # 设置全局模型参数（如果提供）。
    # 调用模型训练器的 io_pfl 方法进行测试时间自适应的渐进联邦学习（PFL），返回测试指标。
    def test_time_adaptation_by_iopfl(self, w_global):
        # 如果传入了全局模型参数 w_global，则设置模型参数为该全局模型参数。这确保模型从一个经过训练的状态开始，而不是随机初始化的状态
        if w_global != None:
            self.model_trainer.set_model_params(w_global)
        metrics = self.model_trainer.io_pfl(self.local_test_data, self.device, self.args)
        return metrics
