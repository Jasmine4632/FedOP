import copy
import logging
import random
import sys, os
import time 
import numpy as np
import pandas as pd
import torch
from .clientditto import ClientDitto

class Ditto:
    def __init__(self, dataset, device, args, model_trainer):
        """
        初始化 Ditto 类
        """
        self.device = device
        self.args = args
        client_num, [
            train_data_num, val_data_num, test_data_num,
            train_data_local_num_dict, train_data_local_dict,
            val_data_local_dict, test_data_local_dict, ood_data
        ] = dataset

        # 客户端数量和数据分布信息
        self.client_num_in_total = client_num
        self.client_num_per_round = int(self.client_num_in_total * self.args.percent)
        self.train_data_num_in_total = train_data_num
        self.val_data_num_in_total = val_data_num
        self.test_data_num_in_total = test_data_num

        # 数据字典
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.val_data_local_dict = val_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.ood_data = ood_data

        # 模型训练器
        self.model_trainer = model_trainer
        self.ood_client = ClientDitto(-1, None, None, ood_data, len(ood_data.dataset), self.args, self.device, model_trainer)
        
        # 客户端列表
        self.client_list = []
        self._setup_clients()  # 初始化客户端

        # 记录本地性能的字典
        self.local_performance_by_global_model = {f'idx{idx}': [] for idx in range(client_num)}
        self.local_val_by_global_model = {f'idx{idx}': [] for idx in range(client_num)}
        self.ood_performance = {"before": []}  # OOD 测试性能记录

    def _setup_clients(self):
        """
        初始化客户端列表，创建每个客户端对应的 ClientDitto 对象
        """
        logging.info("############ Setup Clients ############")
        for client_idx in range(self.client_num_in_total):
            client = ClientDitto(
                client_idx, self.train_data_local_dict[client_idx],
                self.val_data_local_dict[client_idx], self.test_data_local_dict[client_idx],
                self.train_data_local_num_dict[client_idx], self.args, self.device, self.model_trainer
            )
            self.client_list.append(client)

    def train(self):
        """
        训练主流程，包含全局模型聚合与个性化模型训练
        """
        start_time = time.time()
        w_global = self.model_trainer.get_model_params()  # 获取全局模型初始参数

        for round_idx in range(self.args.comm_round):
            logging.info(f"============ Communication round : {round_idx} ============")

            w_locals = []
            client_indexes = self._client_sampling(round_idx)
            logging.info(f"Selected client indexes = {client_indexes}")

            # 遍历每个选中的客户端，进行个性化和全局训练
            for client_idx in client_indexes:
                client = self.client_list[client_idx]
                client.update_local_dataset(
                    client_idx, self.train_data_local_dict[client_idx],
                    self.val_data_local_dict[client_idx], self.test_data_local_dict[client_idx],
                    self.train_data_local_num_dict[client_idx]
                )
                # 训练全局模型和个性化模型
                updated_global_params = client.train_with_personalization(copy.deepcopy(w_global))
                w_locals.append((client.get_sample_number(), updated_global_params))

            # 聚合全局模型参数
            w_global = self._aggregate(w_locals)
            torch.save(w_global, os.path.join(self.args.save_path, "{}_global_round{}".format(self.args.mode, round_idx)))
            self.model_trainer.set_model_params(w_global)

            # 本地验证与测试
            self._local_val_on_all_clients(round_idx)
            self._local_test_on_all_clients(round_idx)

            end_time = time.time()
            logging.info(f"训练完成！总耗时: {end_time - start_time:.2f} 秒")

    def _client_sampling(self, round_idx):
        """
        随机选择客户端参与当前轮次的训练
        """
        if self.client_num_in_total == self.client_num_per_round:
            return list(range(self.client_num_in_total))
        else:
            np.random.seed(round_idx)
            return np.random.choice(range(self.client_num_in_total), self.client_num_per_round, replace=False).tolist()

    def _aggregate(self, w_locals):
        """
        全局模型参数聚合
        """
        training_num = sum([sample_num for sample_num, _ in w_locals])
        aggregated_params = copy.deepcopy(w_locals[0][1])
        for key in aggregated_params.keys():
            aggregated_params[key] = sum(
                local_model_params[key] * (sample_num / training_num)
                for sample_num, local_model_params in w_locals
            )
        return aggregated_params

    def _local_val_on_all_clients(self, round_idx):
        """
        本地验证：在所有客户端上验证个性化模型
        """
        logging.info(f"============ Local Validation : {round_idx} ============")
        for client_idx in range(self.client_num_in_total):
            client = self.client_list[client_idx]
            metrics = client.local_validate(model_type="personal")
            self.local_val_by_global_model[f'idx{client_idx}'].append(metrics['test_acc'])
            logging.info(f"Client {client_idx} - Acc: {metrics['test_acc']:.4f}, Loss: {metrics['test_loss']:.4f}")

    def _local_test_on_all_clients(self, round_idx):
        """
        本地测试：在所有客户端上测试个性化模型
        """
        logging.info(f"============ Local Test : {round_idx} ============")
        for client_idx in range(self.client_num_in_total):
            client = self.client_list[client_idx]
            metrics = client.local_test(b_use_test_dataset=True, model_type="personal")
            self.local_performance_by_global_model[f'idx{client_idx}'].append(metrics['test_acc'])
            logging.info(f"Client {client_idx} - Acc: {metrics['test_acc']:.4f}, Loss: {metrics['test_loss']:.4f}")
    
    def _ood_test_on_global_model(self, round_idx, ood_client, ood_data, w_global):
        logging.info("============ ood_test_global : {}".format(round_idx))
        metrics = ood_client.ood_test(ood_data, w_global)
        test_acc = metrics["test_acc"]
        test_loss = metrics["test_loss"]
        stats = {'test_acc': '{:.4f}'.format(test_acc), 'test_loss': '{:.4f}'.format(test_loss)}
        self.ood_performance['before'].append(test_acc)
        logging.info(stats)
        return metrics