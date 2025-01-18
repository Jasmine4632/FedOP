import copy
import logging
import random
import sys, os
import time
import numpy as np
import pandas as pd
import torch
from .clientapple import ClientAPPLE

# APPLEAPI 类负责实现个性化联邦学习算法 APPLE
class APPLEAPI(object):
    def __init__(self, dataset, device, args, model_trainer):
        """
        dataset: data loaders and data size info
        """
        self.device = device
        self.args = args
        client_num, [train_data_num, val_data_num, test_data_num, train_data_local_num_dict, train_data_local_dict,
                     val_data_local_dict, test_data_local_dict, ood_data] = dataset
        self.client_num_in_total = client_num
        self.client_num_per_round = int(self.client_num_in_total * self.args.percent)
        self.train_data_num_in_total = train_data_num
        self.val_data_num_in_total = val_data_num
        self.test_data_num_in_total = test_data_num

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.val_data_local_dict = val_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.ood_data = ood_data

        self.model_trainer = model_trainer
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, val_data_local_dict, test_data_local_dict, model_trainer)

        logging.info("############setup ood clients#############")
        self.ood_client = ClientAPPLE(-1, None, None, ood_data, len(ood_data.dataset), self.args, self.device, model_trainer)

        self.ood_performance = {"before": []}
        self.local_performance_by_global_model = dict()
        self.local_val_by_global_model = dict()
        for idx in range(client_num):
            self.local_performance_by_global_model[f'idx{idx}'] = []
            self.local_val_by_global_model[f'idx{idx}'] = []

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, val_data_local_dict, test_data_local_dict, model_trainer):
        logging.info("############setup inner clients#############")
        for client_idx in range(self.client_num_in_total):
            c = ClientAPPLE(
                client_idx,
                train_data_local_dict[client_idx],
                val_data_local_dict[client_idx],
                test_data_local_dict[client_idx],
                train_data_local_num_dict[client_idx],
                self.args,
                self.device,
                model_trainer,
                train_data_local_dict=train_data_local_dict  # 动态推断 num_clients
            )
            self.client_list.append(c)

    def train(self):
        start_time = time.time()
        w_global = self.model_trainer.get_model_params()  # 初始化全局模型参数
        # 初始化全局动态权重
        global_weights = [1.0 / self.client_num_in_total for _ in range(self.client_num_in_total)]

        for round_idx in range(self.args.comm_round):
            logging.info("============ Communication round : {}".format(round_idx))
            selected_clients = self._client_sampling(round_idx, self.client_num_in_total, self.client_num_per_round)
            logging.info("client_indexes = " + str(selected_clients))
            
            w_locals = []  # 在这里初始化 w_locals
            
            for client in self.client_list:
                client.set_dynamic_weights(global_weights)

            for idx, client in enumerate(self.client_list):
                client_idx = selected_clients[idx]
                client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                            self.val_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx])
                # 设置动态权重，如果需要
                client.set_dynamic_weights([1.0 / self.client_num_in_total] * self.client_num_in_total)  # 初始化为均匀分布
                
                w_local = client.train(copy.deepcopy(w_global))  # 修改传递参数为 w_global
                w_locals.append((client.get_sample_number(), w_local))
                


            aggregated_model = self._aggregate(global_weights)
            torch.save(aggregated_model, os.path.join(self.args.save_path, "{}_global_round{}".format(self.args.mode, round_idx)))

            self.model_trainer.set_model_params(aggregated_model)

            self._local_val_on_all_clients(round_idx)
            self._local_test_on_all_clients(round_idx)

            end_time = time.time()
            logging.info(f"训练完成！总耗时: {end_time - start_time:.2f} 秒 ({(end_time - start_time) / 60:.2f} 分钟)")

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            # 当所有客户端都参与时，返回顺序列表
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            # 否则随机采样
            np.random.seed(round_idx)  # 确保可复现
            client_indexes = np.random.choice(range(client_num_in_total), client_num_per_round, replace=False).tolist()
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _aggregate(self, weights):
        # 获取全局模型参数
        aggregated_params = copy.deepcopy(self.model_trainer.get_model_params())
    
        # 确保所有聚合参数初始化为 float 类型
        for key in aggregated_params.keys():
            aggregated_params[key] = aggregated_params[key].float()  # 强制转换为 Float 类型

        # 聚合所有客户端的模型参数
        for client, weight in zip(self.client_list, weights):
            client_params = client.get_model_params()
            for key in aggregated_params.keys():
                # 确保 client_params[key] 和 aggregated_params[key] 都是 float 类型
                aggregated_params[key] += client_params[key].float() * weight
    
        return aggregated_params

    def _local_val_on_all_clients(self, round_idx):
        logging.info("============ local_validation_on_all_clients : {}".format(round_idx))
        for client in self.client_list:
            val_metrics = client.local_validate()
            logging.info(f"Client {client.client_idx}: Acc={val_metrics['test_acc']:.4f}, Loss={val_metrics['test_loss']:.4f}")

    def _local_test_on_all_clients(self, round_idx):
        logging.info("============ local_test_on_all_clients : {}".format(round_idx))
        for client in self.client_list:
            test_metrics = client.local_test(b_use_test_dataset=True)
            logging.info(f"Client {client.client_idx}: Acc={test_metrics['test_acc']:.4f}, Loss={test_metrics['test_loss']:.4f}")

    def _ood_test_on_global_model(self, round_idx, ood_client, ood_data, w_global):
        logging.info("============ ood_test_global : {}".format(round_idx))
        metrics = ood_client.ood_test(ood_data, w_global)
        test_acc = metrics["test_acc"]
        test_loss = metrics["test_loss"]
        stats = {'test_acc': '{:.4f}'.format(test_acc), 'test_loss': '{:.4f}'.format(test_loss)}
        self.ood_performance['before'].append(test_acc)
        logging.info(stats)
        return metrics