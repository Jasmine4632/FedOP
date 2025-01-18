import copy
import logging
import random
import sys, os
import time
import numpy as np
import pandas as pd
import torch
from .clientper import ClientPer

# PerFedAvgAPI 类负责实现个性化联邦平均算法
class PerFedAvg:
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
        # setup clients
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, val_data_local_dict, test_data_local_dict, model_trainer)
        logging.info("############setup ood clients#############")
        # !!!!!! 初始化 OOD 客户端
        self.ood_client = ClientPer(-1, None, None, ood_data, len(ood_data.dataset), self.args, self.device, model_trainer)

        self.ood_performance = {"before": []}
        self.local_performance_by_global_model = dict()
        self.local_val_by_global_model = dict()
        for idx in range(client_num):
            self.local_performance_by_global_model[f'idx{idx}'] = []
            self.local_val_by_global_model[f'idx{idx}'] = []

    # 初始化客户端列表
    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, val_data_local_dict,
                       test_data_local_dict, model_trainer):
        logging.info("############setup inner clients#############")
        for client_idx in range(self.client_num_in_total):
            c = ClientPer(client_idx, train_data_local_dict[client_idx], val_data_local_dict[client_idx],
                       test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer)
            self.client_list.append(c)

    # 训练过程
    def train(self):
        # 记录训练开始时间
        start_time = time.time()
        # 获取全局模型的初始参数
        w_global = self.model_trainer.get_model_params()
        logging.info("Starting communication rounds...")

        for round_idx in range(self.args.comm_round):
            logging.info(f"============ Communication round : {round_idx} ============")

            # 存储客户端的本地模型参数
            w_locals = []
            # 随机采样客户端
            client_indexes = self._client_sampling(round_idx, self.client_num_in_total, self.client_num_per_round)
            logging.info(f"client_indexes = {client_indexes}")

            for client_idx in client_indexes:
                client = self.client_list[client_idx]
                # 使用全局模型初始化客户端的个性化模型，并进行本地训练
                w_personal = client.train_with_personalization(copy.deepcopy(w_global))
                # 收集训练后的个性化模型参数
                w_locals.append((client.get_sample_number(), copy.deepcopy(w_personal)))

            # 服务器端聚合客户端更新的模型参数，更新全局模型
            w_global = self._aggregate(w_locals)
            # 保存全局模型参数
            torch.save(w_global, os.path.join(self.args.save_path, f"{self.args.mode}_global_round{round_idx}.pt"))
            # 将聚合后的全局模型参数设置到服务器端模型中
            self.model_trainer.set_model_params(w_global)

            # 本地验证和测试
            self._local_val_on_all_clients(round_idx)
            self._local_test_on_all_clients(round_idx)

            # 记录训练结束时间
            end_time = time.time()
            total_time = end_time - start_time
            logging.info(f"训练完成！总耗时: {total_time:.2f} 秒 ({total_time / 60:.2f} 分钟)")

    # 随机选择客户端
    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info(f"client_indexes = {client_indexes}")
        return client_indexes

    # 聚合客户端模型参数
    def _aggregate(self, w_locals):
        training_num = sum([sample_num for sample_num, _ in w_locals])
        aggregated_params = copy.deepcopy(w_locals[0][1])
        for k in aggregated_params.keys():
            aggregated_params[k] = sum(local_model_params[k] * (sample_num / training_num)
                                       for sample_num, local_model_params in w_locals)
        return aggregated_params

    # 本地验证
    def _local_val_on_all_clients(self, round_idx):
        logging.info(f"============ Local Validation on All Clients : {round_idx} ============")

        val_metrics = {
            'acc': [],
            'losses': []
        }

        for client_idx in range(self.client_num_in_total):
            if self.val_data_local_dict[client_idx] is None:
                continue
            client = self.client_list[client_idx]
            client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                        self.val_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])
            # 调用客户端的 local_validate 方法
            local_metrics = client.local_validate()
        
            # 记录每个客户端的验证指标
            self.local_val_by_global_model[f'idx{client_idx}'].append(copy.deepcopy(local_metrics['test_acc']))
            val_metrics['acc'].append(copy.deepcopy(local_metrics['test_acc']))
            val_metrics['losses'].append(copy.deepcopy(local_metrics['test_loss']))
        
            logging.info('Client Index = {}\tAcc:{:.4f}\tLoss: {:.4f}'.format(
                client_idx, local_metrics['test_acc'], local_metrics['test_loss'] ))


    # 本地测试
    def _local_test_on_all_clients(self, round_idx):
        logging.info(f"============ Local Testing on All Clients : {round_idx} ============")

        test_metrics = {
            'acc': [],
            'losses': []
        }

        for client_idx in range(self.client_num_in_total):
            if self.test_data_local_dict[client_idx] is None:
                continue
            client = self.client_list[client_idx]
            client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                        self.val_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])
            # 调用客户端的 local_test 方法
            test_local_metrics = client.local_test(True)

            # 记录每个客户端的测试指标
            self.local_performance_by_global_model[f'idx{client_idx}'].append(copy.deepcopy(test_local_metrics['test_acc']))
            test_metrics['acc'].append(copy.deepcopy(test_local_metrics['test_acc']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))
        
            logging.info('Client Index = {}\tAcc:{:.4f}\tLoss: {:.4f}'.format(
                client_idx, test_local_metrics['test_acc'], test_local_metrics['test_loss']))

    def _ood_test_on_global_model(self, round_idx, ood_client, ood_data, w_global):
        logging.info("============ ood_test_global : {}".format(round_idx))
        metrics = ood_client.ood_test(ood_data, w_global)
        test_acc = metrics["test_acc"]
        test_loss = metrics["test_loss"]
        stats = {'test_acc': '{:.4f}'.format(test_acc), 'test_loss': '{:.4f}'.format(test_loss)}
        self.ood_performance['before'].append(test_acc)
        logging.info(stats)
        return metrics