import copy
import logging
import random
import sys, os
import time 
import numpy as np
import pandas as pd
import torch
from .clientiop import ClientIOP

# FedAvgAPI 类负责实现联邦平均算法
class FedIOP(object):
    # __init__ 方法初始化对象，设置数据集、设备、参数和模型训练器
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
        # !!!!!!
        self.ood_client = ClientIOP(-1, None, None, ood_data, len(ood_data.dataset), self.args, self.device, model_trainer)

        self.ood_performance = {"before": []}
        self.local_performance_by_global_model = dict()
        self.local_performance_by_trajectory = dict()
        self.local_val_by_global_model = dict()
        self.local_val_by_trajectory = dict()
        self.ood_performance_by_trajectory = dict()
        for idx in range(client_num):
            self.local_performance_by_global_model[f'idx{idx}'] = []
            self.local_performance_by_trajectory[f'idx{idx}'] = []
            self.ood_performance_by_trajectory[f'idx{idx}'] = []
            self.local_val_by_global_model[f'idx{idx}'] = []
            self.local_val_by_trajectory[f'idx{idx}'] = []

    # setup_clients 方法设置客户端
    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, val_data_local_dict,
                       test_data_local_dict, model_trainer):
        logging.info("############setup inner clients#############")
        for client_idx in range(self.client_num_in_total):
            c = ClientIOP(client_idx, train_data_local_dict[client_idx], val_data_local_dict[client_idx],
                       test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer)
            self.client_list.append(c)
        # logging.info("############setup_clients (END)#############")

    # train 方法执行联邦训练过程，包括客户端训练和模型聚合
    def train(self):
        # 记录训练开始时间
        start_time = time.time()
        # 获取全局模型的初始参数，并存储在 w_global 中
        w_global = self.model_trainer.get_model_params()
        # 开始一个循环，共有 self.args.comm_round 轮，每一轮代表一次通信
        for round_idx in range(self.args.comm_round):
            # 打印当前通信轮次的信息
            logging.info("============ Communication round : {}".format(round_idx))
            # 初始化一个空列表，用于存储各个客户端的本地模型参数
            w_locals = []
            # 随机选择一部分客户端参与当前轮次的训练
            client_indexes = self._client_sampling(round_idx, self.client_num_in_total,
                                                   self.client_num_per_round)
            logging.info("client_indexes = " + str(client_indexes))
            # 更新每个客户端的数据集。
            # 使用全局模型参数在本地数据集上训练客户端模型，并返回训练后的本地模型参数。
            # 将本地模型参数和样本数量添加到 w_locals 列表中
            for idx, client in enumerate(self.client_list):
                # update dataset
                client_idx = client_indexes[idx]
                client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                            self.val_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx])

                # train on new dataset
                w = client.train(copy.deepcopy(w_global))
                client.save_trajectory(round_idx)
                # logging.info("local weights = " + str(w))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

            # 聚合所有客户端的本地模型参数，更新全局模型参数
            w_global = self._aggregate(w_locals)
            # save global weights
            torch.save(w_global, os.path.join(self.args.save_path, "{}_global_round{}".format(self.args.mode, round_idx)))
            # 将聚合后的全局模型参数设置到模型中
            self.model_trainer.set_model_params(w_global)
            # 在所有客户端上进行本地验证和测试，记录模型的性能
            self._local_val_on_all_clients(round_idx)
            # local test
            self._local_test_on_all_clients(round_idx)
            # test results
            # self._ood_test_on_global_model(round_idx, self.ood_client, self.ood_data, w_global)
            # self._ood_test_on_trajectory(round_idx)
            
            # 记录训练结束时间
            end_time = time.time()
            total_time = end_time - start_time  # 计算总时间

            # 打印训练总耗时
            logging.info(f"训练完成！总耗时: {total_time:.2f} 秒 ({total_time / 60:.2f} 分钟)")

            # # local val results  将本地验证和测试结果保存为 CSV 文件
            # local_val_by_global_model_pd = pd.DataFrame.from_dict(self.local_val_by_global_model)
            # local_val_by_trajectory_pd = pd.DataFrame.from_dict(self.local_val_by_trajectory)
            # # local test results
            # local_performance_by_global_model_pd = pd.DataFrame.from_dict(self.local_performance_by_global_model)
            # local_performance_by_trajectory_pd = pd.DataFrame.from_dict(self.local_performance_by_trajectory)
            # ood results
            # ood_performance_pd = pd.DataFrame.from_dict(self.ood_performance)
            # ood_performance_by_trajectory_pd = pd.DataFrame.from_dict(self.ood_performance_by_trajectory)

            # local_val_by_global_model_pd.to_csv(
            #     os.path.join(self.args.save_path, self.args.mode + "_local_val_by_global_model.csv"))
            # local_val_by_trajectory_pd.to_csv(
            #     os.path.join(self.args.save_path, self.args.mode + "_local_val_by_trajectory.csv"))
            # local_performance_by_global_model_pd.to_csv(
            #     os.path.join(self.args.save_path, self.args.mode + "_local_performance_by_global_model.csv"))
            # local_performance_by_trajectory_pd.to_csv(
            #     os.path.join(self.args.save_path, self.args.mode + "_local_performance_by_trajectory.csv"))
            # ood_performance_pd.to_csv(os.path.join(self.args.save_path, self.args.mode + "_ood_performance.csv"))
            # ood_performance_by_trajectory_pd.to_csv(
            #     os.path.join(self.args.save_path, self.args.mode + "_ood_performance_by_trajectory.csv"))

    # sample_clients 方法随机选择客户端参与当前轮次的训练
    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    # aggregate 方法聚合客户端模型参数，更新全局模型
    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    # 测试方法 _ood_test_on_global_model 和 _ood_test_on_trajectory
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

    # 在轨迹模型上进行 OOD 测试
    def _ood_test_on_trajectory(self, round_idx):
        logging.info("============ ood_test_on_all_trajectory : {}".format(round_idx))
        for client_idx in range(self.client_num_in_total):
            client = self.client_list[client_idx]
            test_ood_metrics_by_trajectory = client.ood_test_by_trajectory(self.ood_data)
            self.ood_performance_by_trajectory["idx" + str(client_idx)].append(
                copy.deepcopy(test_ood_metrics_by_trajectory['test_acc']))

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
            client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                        self.val_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])
            local_metrics = client.local_validate()
            local_metrics_by_trajectory = client.local_validate_by_trajectory()

            self.local_val_by_global_model["idx" + str(client_idx)].append(copy.deepcopy(local_metrics['test_acc']))
            self.local_val_by_trajectory["idx" + str(client_idx)].append(copy.deepcopy(local_metrics_by_trajectory['test_acc']))
            val_metrics['acc'].append(copy.deepcopy(local_metrics['test_acc']))
            val_metrics['losses'].append(copy.deepcopy(local_metrics_by_trajectory['test_loss']))
            logging.info('Client Index = {}\tAcc:{:.4f}\tLoss: {:.4f}'.format(
                client_idx, local_metrics['test_acc'], local_metrics_by_trajectory['test_loss']))
        # logging.info(val_metrics)

    # 在所有客户端上进行本地测试，记录性能
    def _local_test_on_all_clients(self, round_idx):
        logging.info("============ local_test_on_all_clients : {}".format(round_idx))

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
            # test data
            test_local_metrics = client.local_test(True)
            test_local_metrics_by_trajectory = client.local_test_by_trajectory()

            self.local_performance_by_global_model["idx" + str(client_idx)].append(
                copy.deepcopy(test_local_metrics['test_acc']))
            self.local_performance_by_trajectory["idx" + str(client_idx)].append(
                copy.deepcopy(test_local_metrics_by_trajectory['test_acc']))
            test_metrics['acc'].append(copy.deepcopy(test_local_metrics['test_acc']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))
            logging.info('Client Index = {}\tAcc:{:.4f}\tLoss: {:.4f}'.format(
                client_idx, test_local_metrics['test_acc'], test_local_metrics['test_loss']))
        # logging.info(test_metrics)

    # 测试时间自适应方法 test_time_adaptation 在测试时进行模型自适应，记录性能
    def test_time_adaptation(self, w_global=None):
        metrics = self.ood_client.test_time_adaptation_by_iopfl(copy.deepcopy(w_global))

        test_acc = metrics["test_acc"]
        test_loss = metrics["test_loss"]
        self.ood_performance["after"].append(test_acc)
        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        logging.info("############  performance after test time adaptation  #############")
        logging.info(stats)
        return metrics





