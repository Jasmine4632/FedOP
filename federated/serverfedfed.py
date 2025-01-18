import copy
import logging
import random
import sys, os
import time
import numpy as np
import pandas as pd
import torch
from .clientfedfed import ClientFedFed

class FedFed:
    def __init__(self, dataset, device, args, model_trainer):
        """
        初始化 FedFed 算法
        """
        self.device = device
        self.args = args

        # 解压数据集
        client_num, [train_data_num, val_data_num, test_data_num, train_data_local_num_dict, train_data_local_dict,
                     val_data_local_dict, test_data_local_dict, ood_data] = dataset
        self.client_num_in_total = client_num
        self.client_num_per_round = int(self.client_num_in_total * self.args.percent)
        self.train_data_num_in_total = train_data_num
        self.val_data_num_in_total = val_data_num
        self.test_data_num_in_total = test_data_num
        self.ood_data = ood_data
        # 客户端相关数据
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.val_data_local_dict = val_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        # 初始化模型
        self.model_trainer = model_trainer
        self.global_model_params = self.model_trainer.get_model_params()
        self.ood_client = ClientFedFed(-1, None, None, ood_data, len(ood_data.dataset), self.args, self.device, model_trainer)

        # 设置客户端
        self._setup_clients()

        # 初始化性能记录
        self.ood_performance = {"before": []}
        self.local_performance_by_global_model = dict()
        self.local_val_by_global_model = dict()
        for idx in range(client_num):
            self.local_performance_by_global_model[f'idx{idx}'] = []
            self.local_val_by_global_model[f'idx{idx}'] = []

    def _setup_clients(self):
        """
        初始化客户端
        """
        logging.info("############ Setup Clients ############")
        for client_idx in range(self.client_num_in_total):
            client = ClientFedFed(
                client_idx,
                self.train_data_local_dict[client_idx],
                self.val_data_local_dict[client_idx],
                self.test_data_local_dict[client_idx],
                self.train_data_local_num_dict[client_idx],
                self.args,
                self.device,
                self.model_trainer
            )
            self.client_list.append(client)

    def train(self):
        """
        执行联邦训练
        """
        # 记录训练开始时间
        start_time = time.time()
        # 获取全局模型的初始参数，并存储在 w_global 中
        w_global = self.model_trainer.get_model_params()
        logging.info("Starting communication rounds...")

        # 开始一个循环，共有 self.args.comm_round 轮，每一轮代表一次通信
        for round_idx in range(self.args.comm_round):
            # 打印当前通信轮次的信息
            logging.info("============ Communication round : {} ============".format(round_idx))

            # 初始化一个空列表，用于存储各个客户端的本地模型参数
            w_locals = []
            # 随机选择一部分客户端参与当前轮次的训练
            client_indexes = self._client_sampling(round_idx, self.client_num_in_total, self.client_num_per_round)
            logging.info("client_indexes = {}".format(client_indexes))

            # 更新每个客户端的数据集并进行训练
            for idx in range(len(client_indexes)):
                client_idx = client_indexes[idx]
                client = self.client_list[client_idx]
                client.update_local_dataset(
                    client_idx, 
                    self.train_data_local_dict[client_idx],
                    self.val_data_local_dict[client_idx],
                    self.test_data_local_dict[client_idx],
                    self.train_data_local_num_dict[client_idx]
                )

                # 在本地数据上训练客户端模型，并返回本地模型参数
                w = client.train(copy.deepcopy(w_global))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

            # 聚合所有客户端的本地模型参数，更新全局模型参数
            w_global = self._aggregate(w_locals)

            # 保存当前通信轮次的全局模型
            torch.save(
                w_global, 
                os.path.join(self.args.save_path, "{}_global_round{}".format(self.args.mode, round_idx))
            )

            # 将聚合后的全局模型参数设置到模型中
            self.model_trainer.set_model_params(w_global)

            # 在所有客户端上进行本地验证
            self._local_val_on_all_clients(round_idx)

            # 在所有客户端上进行本地测试
            self._local_test_on_all_clients(round_idx)

            # 记录训练结束时间
            end_time = time.time()
            total_time = end_time - start_time  # 计算总时间

            # 打印训练总耗时
            logging.info(f"训练完成！总耗时: {total_time:.2f} 秒 ({total_time / 60:.2f} 分钟)")

    def _ood_test_on_global_model(self, round_idx, ood_client, ood_data, w_global):
        logging.info("============ ood_test_global : {}".format(round_idx))
        metrics = ood_client.ood_test(ood_data, w_global)
        ''' unify key'''
        test_acc = metrics["test_acc"]
        test_loss = metrics["test_loss"]
        stats = {'test_acc': '{:.4f}'.format(test_acc), 'test_loss': '{:.4f}'.format(test_loss)}
        self.ood_performance['before'].append(test_acc)            
        logging.info(stats)
        return metrics
    
    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _aggregate(self, w_locals):
        """
        聚合客户端模型参数
        """
        training_num = sum([local[0] for local in w_locals])
        averaged_params = copy.deepcopy(w_locals[0][1])

        for k in averaged_params.keys():
            averaged_params[k] = sum(local[1][k] * (local[0] / training_num) for local in w_locals)
        return averaged_params

    def _local_test_on_all_clients(self, round_idx):
        """
        在所有客户端上进行本地测试
        """
        logging.info(f"============ Local testing on all clients : {round_idx} ============")

        for client_idx, client in enumerate(self.client_list):
            if client.local_test_data is None:
                logging.warning(f"Client {client_idx} has no test data.")
                continue

            # 本地测试
            test_metrics = client.local_test(True)
            self.local_performance_by_global_model[f'idx{client_idx}'].append(test_metrics['test_acc'])

            logging.info(f"Client {client_idx} test - Acc: {test_metrics['test_acc']:.4f}, Loss: {test_metrics['test_loss']:.4f}")


    def _local_val_on_all_clients(self, round_idx):
        """
        在所有客户端上进行本地验证
        """
        logging.info(f"============ Local validation on all clients : {round_idx} ============")

        for client_idx, client in enumerate(self.client_list):
            if client.local_val_data is None:
                logging.warning(f"Client {client_idx} has no validation data.")
                continue

            # 本地验证
            val_metrics = client.local_validate()
            self.local_val_by_global_model[f'idx{client_idx}'].append(val_metrics['test_acc'])

            logging.info(f"Client {client_idx} validation - Acc: {val_metrics['test_acc']:.4f}, Loss: {val_metrics['test_loss']:.4f}")
