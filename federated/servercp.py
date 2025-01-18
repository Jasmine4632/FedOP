import copy
import logging
import os
import time
import numpy as np
import torch
from .clientcp import ClientCP


class FedCP:
    def __init__(self, dataset, device, args, model_trainer):
        """
        初始化 FedCP
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

        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.val_data_local_dict = val_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.ood_data = ood_data

        self.model_trainer = model_trainer
        self.client_list = []

        # 初始化客户端
        self._setup_clients()
        logging.info("############ Setup OOD Client ############")
        self.ood_client = ClientCP(
            -1, None, None, ood_data, len(ood_data.dataset), self.args, self.device, model_trainer
        )

        # 用于记录客户端的历史性能指标
        self.client_metrics = {f"client_{i}": {"accuracy": [], "loss": []} for i in range(client_num)}
        self.ood_performance = {"before": []}
        self.local_performance_by_global_model = dict()
        self.local_val_by_global_model = dict()
        for idx in range(client_num):
            self.local_performance_by_global_model[f'idx{idx}'] = []
            self.local_val_by_global_model[f'idx{idx}'] = []        

    def _setup_clients(self):
        """
        初始化所有客户端
        """
        logging.info("############ Setup Inner Clients ############")
        for client_idx in range(self.client_num_in_total):
            client = ClientCP(
                client_idx,
                self.train_data_local_dict[client_idx],
                self.val_data_local_dict[client_idx],
                self.test_data_local_dict[client_idx],
                self.train_data_local_num_dict[client_idx],
                self.args,
                self.device,
                self.model_trainer,
            )
            self.client_list.append(client)

    def train(self):
        """
        执行 FedCP 联邦训练
        """
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

            # 条件客户端采样
            client_indexes = self._conditional_sampling(round_idx)
            logging.info("client_indexes = " + str(client_indexes))

            # 更新每个客户端的数据集
            for idx, client in enumerate(self.client_list):
                if idx not in client_indexes:
                    continue

                client_idx = client_indexes[idx]
                client.update_local_dataset(
                    client_idx,
                    self.train_data_local_dict[client_idx],
                    self.val_data_local_dict[client_idx],
                    self.test_data_local_dict[client_idx],
                    self.train_data_local_num_dict[client_idx]
                )

                # 使用全局模型参数在本地数据集上训练客户端模型，并返回训练后的本地模型参数
                w = client.train(copy.deepcopy(w_global))

                # 将本地模型参数和样本数量添加到 w_locals 列表中
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

            # 聚合所有客户端的本地模型参数，更新全局模型参数
            w_global = self._aggregate(w_locals)

            # 保存全局模型权重
            torch.save(w_global, os.path.join(self.args.save_path, f"{self.args.mode}_global_round{round_idx}"))

            # 将聚合后的全局模型参数设置到模型中
            self.model_trainer.set_model_params(w_global)

            # 在所有客户端上进行本地验证和测试，记录模型的性能
            self._local_val_on_all_clients(round_idx)
            self._local_test_on_all_clients(round_idx)

        # 记录训练结束时间
        end_time = time.time()
        total_time = end_time - start_time  # 计算总时间

        # 打印训练总耗时
        logging.info(f"训练完成！总耗时: {total_time:.2f} 秒 ({total_time / 60:.2f} 分钟)")

    def _conditional_sampling(self, round_idx):
        """
        基于客户端性能的条件采样
        """
        participation_probabilities = []
        for client_idx, metrics in self.client_metrics.items():
            if metrics["accuracy"]:  # 有性能数据时
                recent_acc = np.mean(metrics["accuracy"][-5:])  # 最近5轮平均准确率
                prob = 1 / (1 + np.exp(-recent_acc))  # Sigmoid 激活函数
            else:
                prob = 0.5  # 默认初始概率
            participation_probabilities.append(prob)

        # Normalize probabilities
        probabilities = np.array(participation_probabilities)
        probabilities /= probabilities.sum()

        # Random sampling based on probabilities
        np.random.seed(round_idx)
        selected_clients = np.random.choice(
            range(self.client_num_in_total),
            self.client_num_per_round,
            replace=False,
            p=probabilities
        )

        logging.info(f"Selected clients: {selected_clients}")
        return selected_clients

    def _aggregate(self, w_locals):
        """
        聚合客户端模型参数
        """
        training_num = sum([num_samples for num_samples, _ in w_locals])
        aggregated_params = copy.deepcopy(w_locals[0][1])

        for k in aggregated_params.keys():
            aggregated_params[k] = aggregated_params[k].float()  # 确保参数为浮点类型
            aggregated_params[k] *= w_locals[0][0] / training_num
            for i in range(1, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                aggregated_params[k] += local_model_params[k].float() * (local_sample_number / training_num)

        return aggregated_params

    def _local_val_on_all_clients(self, round_idx):
        """
        本地验证
        """
        logging.info(f"============ Local Validation on All Clients : {round_idx} ============")
        for client_idx in range(self.client_num_in_total):
            client = self.client_list[client_idx]
            metrics = client.local_validate()
            self.client_metrics[f"client_{client_idx}"]["accuracy"].append(metrics['test_acc'])
            self.client_metrics[f"client_{client_idx}"]["loss"].append(metrics['test_loss'])
            logging.info(f"Client {client_idx} Validation - Acc: {metrics['test_acc']:.4f}, Loss: {metrics['test_loss']:.4f}")

    def _local_test_on_all_clients(self, round_idx):
        """
        本地测试
        """
        logging.info(f"============ Local Test on All Clients : {round_idx} ============")
        for client_idx in range(self.client_num_in_total):
            client = self.client_list[client_idx]
            metrics = client.local_test(True)
            logging.info(f"Client {client_idx} Test - Acc: {metrics['test_acc']:.4f}, Loss: {metrics['test_loss']:.4f}")
            
    def _ood_test_on_global_model(self, round_idx, ood_client, ood_data, w_global):
        logging.info("============ ood_test_global : {}".format(round_idx))
        metrics = ood_client.ood_test(ood_data, w_global)
        test_acc = metrics["test_acc"]
        test_loss = metrics["test_loss"]
        stats = {'test_acc': '{:.4f}'.format(test_acc), 'test_loss': '{:.4f}'.format(test_loss)}
        self.ood_performance['before'].append(test_acc)
        logging.info(stats)
        return metrics