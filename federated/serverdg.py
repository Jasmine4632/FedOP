import copy
import logging
import random
import sys, os
import time 
import numpy as np
import pandas as pd
import torch
from .clientdg import ClientDG  # 确保路径正确

# FedDG 类负责实现联邦领域泛化算法
class FedDG(object):
    # __init__ 方法初始化对象，设置数据集、设备、参数和模型训练器
    def __init__(self, dataset, device, args, model_trainer):
        """
        dataset: 数据加载器和数据大小信息，格式为：
            (
                client_num,
                [
                    train_data_num,
                    val_data_num,
                    test_data_num,
                    train_data_local_num_dict,
                    train_data_local_dict,
                    val_data_local_dict,
                    test_data_local_dict,
                    ood_data
                ]
            )
        device: 训练设备
        args: 配置参数
        model_trainer: 模型训练器
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
        # Initialize the OOD client
        self.ood_client = ClientDG(
            client_idx=-1,
            local_training_data=None,
            local_val_data=None,
            local_test_data=None,
            local_sample_number=0,
            ood_data=ood_data,
            args=self.args,
            device=self.device,
            model_trainer=model_trainer
        )

        self.ood_performance = {"before": []}
        self.local_performance_by_global_model = dict()
        self.local_val_by_global_model = dict()
        for idx in range(client_num):
            self.local_performance_by_global_model[f'idx{idx}'] = []
            self.local_val_by_global_model[f'idx{idx}'] = []

    # setup_clients 方法设置客户端
    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, val_data_local_dict,
                       test_data_local_dict, model_trainer):
        logging.info("############setup inner clients#############")
        for client_idx in range(self.client_num_in_total):
            c = ClientDG(
                client_idx, 
                train_data_local_dict[client_idx], 
                val_data_local_dict[client_idx],
                test_data_local_dict[client_idx],
                train_data_local_num_dict[client_idx], 
                self.ood_data,        # 传递 ood_data
                self.args, 
                self.device, 
                model_trainer
            )
            self.client_list.append(c)

    # train 方法执行联邦训练过程，包括客户端训练和模型聚合
    def train(self):
        # 记录训练开始时间
        start_time = time.time()
        # 获取全局模型的初始参数，并存储在 w_global 中
        w_global = self.model_trainer.get_model_params()
        logging.info("Starting communication rounds...")

        # 开始一个循环，共有 self.args.comm_round 轮，每一轮代表一次通信
        for round_idx in range(self.args.comm_round):
            # 打印当前通信轮次的信息
            logging.info("============ Communication round : {}".format(round_idx))

            # 初始化一个空列表，用于存储各个客户端的本地模型参数
            w_locals = []
            # 随机选择一部分客户端参与当前轮次的训练
            client_indexes = self._client_sampling(round_idx, self.client_num_in_total, self.client_num_per_round)
            logging.info("client_indexes = " + str(client_indexes))
            # 更新每个客户端的数据集
            for idx, client in enumerate(self.client_list):
                client_idx = client_indexes[idx]
                client.update_local_dataset(
                    client_idx, 
                    self.train_data_local_dict[client_idx],
                    self.val_data_local_dict[client_idx],
                    self.test_data_local_dict[client_idx],
                    self.train_data_local_num_dict[client_idx],
                    self.ood_data  # 传递 ood_data
                )

                # 使用全局模型参数在本地数据集上训练客户端模型，并返回训练后的本地模型参数。
                w = client.train(copy.deepcopy(w_global))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

            # 聚合所有客户端的本地模型参数，更新全局模型参数
            w_global = self._aggregate(w_locals)
            # 保存全局模型权重
            torch.save(w_global, os.path.join(self.args.save_path, "{}_global_round{}".format(self.args.mode, round_idx)))
            # 将聚合后的全局模型参数设置到模型中
            self.model_trainer.set_model_params(w_global)
            # 在所有客户端上进行本地验证和测试，记录模型的性能
            self._local_val_on_all_clients(round_idx)
            self._local_test_on_all_clients(round_idx)
            # 在OOD数据上测试
            # self._ood_test_on_global_model(round_idx, self.ood_client, self.ood_data, w_global)
            
            # 记录训练结束时间
            end_time = time.time()
            total_time = end_time - start_time  # 计算总时间
            logging.info(f"训练完成！总耗时: {total_time:.2f} 秒 ({total_time / 60:.2f} 分钟)")

    # sample_clients 方法随机选择客户端参与当前轮次的训练
    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # ensure same clients for each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    # aggregate 方法聚合客户端模型参数，更新全局模型
    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        # 初始化聚合参数
        aggregated_params = copy.deepcopy(w_locals[0][1])
        for k in aggregated_params.keys():
            aggregated_params[k] = 0  # 初始化为0

        # 聚合所有客户端的参数
        for sample_num, local_params in w_locals:
            weight = sample_num / training_num
            for k in aggregated_params.keys():
                aggregated_params[k] += local_params[k] * weight

        return aggregated_params

    # OOD 测试方法 _ood_test_on_global_model 
    # 在全局模型上进行 OOD 测试
    def _ood_test_on_global_model(self, round_idx, ood_client, ood_data, w_global):
        logging.info("============ ood_test_global : {}".format(round_idx))
        metrics = ood_client.ood_test(ood_data, w_global)
        test_acc = metrics["test_acc"]
        test_loss = metrics["test_loss"]
        stats = {'test_acc': '{:.4f}'.format(test_acc), 'test_loss': '{:.4f}'.format(test_loss)}
        self.ood_performance['before'].append(test_acc)
        logging.info(stats)
        return metrics

    # 本地验证和测试方法 _local_val_on_all_clients 和 _local_test_on_all_clients
    # 在所有客户端上进行本地验证和测试，记录性能
    def _local_val_on_all_clients(self, round_idx):
        logging.info("============ local_validation_on_all_clients : {}".format(round_idx))

        val_metrics = {
            'acc': [],
            'losses': []
        }

        for client_idx in range(self.client_num_in_total):
            if self.val_data_local_dict[client_idx] is None:
                continue
            client = self.client_list[client_idx]
            client.update_local_dataset(
                client_idx, 
                self.train_data_local_dict[client_idx],
                self.val_data_local_dict[client_idx],
                self.test_data_local_dict[client_idx],
                self.train_data_local_num_dict[client_idx],
                self.ood_data  # 传递 ood_data
            )

            local_metrics = client.local_validate()

            self.local_val_by_global_model["idx" + str(client_idx)].append(copy.deepcopy(local_metrics['test_acc']))
            val_metrics['acc'].append(copy.deepcopy(local_metrics['test_acc']))
            val_metrics['losses'].append(copy.deepcopy(local_metrics['test_loss']))
            logging.info('Client Index = {}\tAcc:{:.4f}\tLoss: {:.4f}'.format(
                client_idx, local_metrics['test_acc'], local_metrics['test_loss'] ))
        # logging.info(val_metrics)

    def _local_test_on_all_clients(self, round_idx):
        logging.info("============ local_testing_on_all_clients : {}".format(round_idx))

        test_metrics = {
            'acc': [],
            'losses': []
        }

        for client_idx in range(self.client_num_in_total):
            if self.test_data_local_dict[client_idx] is None:
                continue
            client = self.client_list[client_idx]
            client.update_local_dataset(
                client_idx, 
                self.train_data_local_dict[client_idx],
                self.val_data_local_dict[client_idx],
                self.test_data_local_dict[client_idx],
                self.train_data_local_num_dict[client_idx],
                self.ood_data  # 传递 ood_data
            )

            # 使用测试数据进行测试
            local_metrics = client.local_test(b_use_test_dataset=True)

            self.local_performance_by_global_model[f"idx{client_idx}"].append(copy.deepcopy(local_metrics['test_acc']))
            test_metrics['acc'].append(copy.deepcopy(local_metrics['test_acc']))
            test_metrics['losses'].append(copy.deepcopy(local_metrics['test_loss']))
            logging.info('Client Index = {}\tAcc:{:.4f}\tLoss: {:.4f}'.format(
                client_idx, local_metrics['test_acc'], local_metrics['test_loss']
            ))
        # logging.info(test_metrics)
