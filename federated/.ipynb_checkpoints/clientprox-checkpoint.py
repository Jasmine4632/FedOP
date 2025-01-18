import copy
import logging
import random
import sys, os
import time
import numpy as np
import torch
from .clientprox import ClientProx

class FedProx:
    def __init__(self, dataset, device, args, model_trainer):
        """
        初始化 FedProx
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
        self._setup_clients()
        logging.info("############setup ood clients#############")
        self.ood_client = ClientProx(-1, None, None, ood_data, len(ood_data.dataset), self.args, self.device, model_trainer)

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
        logging.info("############setup inner clients#############")
        for client_idx in range(self.client_num_in_total):
            c = ClientProx(client_idx, self.train_data_local_dict[client_idx], self.val_data_local_dict[client_idx],
                           self.test_data_local_dict[client_idx], self.train_data_local_num_dict[client_idx], self.args,
                           self.device, self.model_trainer)
            self.client_list.append(c)

    def train(self):
        """
        执行联邦训练过程
        """
        start_time = time.time()
        w_global = self.model_trainer.get_model_params()
        logging.info("Starting communication rounds...")

        for round_idx in range(self.args.comm_round):
            logging.info("============ Communication round : {}".format(round_idx))
            w_locals = []
            client_indexes = self._client_sampling(round_idx, self.client_num_in_total, self.client_num_per_round)
            logging.info("client_indexes = " + str(client_indexes))

            for idx, client in enumerate(self.client_list):
                client_idx = client_indexes[idx]
                client.update_local_dataset(
                    client_idx, self.train_data_local_dict[client_idx],
                    self.val_data_local_dict[client_idx],
                    self.test_data_local_dict[client_idx],
                    self.train_data_local_num_dict[client_idx]
                )
                w = client.train(copy.deepcopy(w_global))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

            w_global = self._aggregate(w_locals)
            torch.save(w_global, os.path.join(self.args.save_path, "{}_global_round{}".format(self.args.mode, round_idx)))
            self.model_trainer.set_model_params(w_global)
            self._local_val_on_all_clients(round_idx)
            self._local_test_on_all_clients(round_idx)

        end_time = time.time()
        logging.info(f"训练完成！总耗时: {end_time - start_time:.2f} 秒 ({(end_time - start_time) / 60:.2f} 分钟)")

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

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

    def _local_val_on_all_clients(self, round_idx):
        logging.info("============ local_validation_on_all_clients : {}".format(round_idx))
        for client_idx in range(self.client_num_in_total):
            if self.val_data_local_dict[client_idx] is None:
                continue
            client = self.client_list[client_idx]
            client.update_local_dataset(
                client_idx, self.train_data_local_dict[client_idx],
                self.val_data_local_dict[client_idx],
                self.test_data_local_dict[client_idx],
                self.train_data_local_num_dict[client_idx]
            )
            local_metrics = client.local_validate()
            self.local_val_by_global_model["idx" + str(client_idx)].append(local_metrics['test_acc'])

    def _local_test_on_all_clients(self, round_idx):
        logging.info("============ local_test_on_all_clients : {}".format(round_idx))
        for client_idx in range(self.client_num_in_total):
            if self.test_data_local_dict[client_idx] is None:
                continue
            client = self.client_list[client_idx]
            client.update_local_dataset(
                client_idx, self.train_data_local_dict[client_idx],
                self.val_data_local_dict[client_idx],
                self.test_data_local_dict[client_idx],
                self.train_data_local_num_dict[client_idx]
            )
            test_local_metrics = client.local_test(True)
            self.local_performance_by_global_model["idx" + str(client_idx)].append(test_local_metrics['test_acc'])
