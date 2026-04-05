import copy
import logging
import os
import time
from collections import OrderedDict

import numpy as np
import torch

from .clientfednca import ClientFedNCA


def weighted_average_partial(states, param_names):
    if len(states) == 0:
        raise ValueError('states must not be empty')
    total = float(sum(n for n, _ in states))
    avg = OrderedDict()
    for name in param_names:
        avg[name] = torch.zeros_like(states[0][1][name].detach().cpu())
    for n, st in states:
        w = n / total
        for name in param_names:
            avg[name] += st[name].detach().cpu() * w
    return avg


class FedNCAAPI(object):
    def __init__(self, dataset, device, args, model_trainer):
        self.device = device
        self.args = args
        client_num, [train_data_num, val_data_num, test_data_num,
                     train_data_local_num_dict, train_data_local_dict,
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
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, val_data_local_dict, test_data_local_dict, model_trainer)
        self.ood_client = ClientFedNCA(-1, None, None, ood_data, len(ood_data.dataset), self.args, self.device, model_trainer)
        self.local_performance_by_global_model = {f'idx{i}': [] for i in range(client_num)}
        self.local_val_by_global_model = {f'idx{i}': [] for i in range(client_num)}

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, val_data_local_dict, test_data_local_dict, model_trainer):
        logging.info('############ setup FedNCA inner clients #############')
        for client_idx in range(self.client_num_in_total):
            client = ClientFedNCA(client_idx, train_data_local_dict[client_idx], val_data_local_dict[client_idx],
                                  test_data_local_dict[client_idx], train_data_local_num_dict[client_idx],
                                  self.args, self.device, model_trainer)
            self.client_list.append(client)

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            return [cid for cid in range(client_num_in_total)]
        np.random.seed(round_idx)
        return np.random.choice(range(client_num_in_total), min(client_num_per_round, client_num_in_total), replace=False)

    def train(self):
        start_time = time.time()
        w_global = self.model_trainer.get_model_params()
        logging.info('Starting FedNCA communication rounds...')
        for round_idx in range(self.args.comm_round):
            logging.info('============ FedNCA Communication round : {}'.format(round_idx))
            client_indexes = self._client_sampling(round_idx, self.client_num_in_total, self.client_num_per_round)
            local_partial_states = []
            selected_names = None
            for idx, client in enumerate(self.client_list[:len(client_indexes)]):
                client_idx = client_indexes[idx]
                client.update_local_dataset(client_idx,
                                            self.train_data_local_dict[client_idx],
                                            self.val_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx])
                result = client.train(copy.deepcopy(w_global))
                selected_names = result['selected_names']
                local_partial_states.append((client.get_sample_number(), result['partial']))
            if selected_names:
                avg_partial = weighted_average_partial(local_partial_states, selected_names)
                for name in selected_names:
                    w_global[name] = avg_partial[name]
            torch.save(w_global, os.path.join(self.args.save_path, f'{self.args.mode}_global_round{round_idx}'))
            self.model_trainer.set_model_params(w_global)
            self._local_val_on_all_clients()
            self._local_test_on_all_clients()
        elapsed = time.time() - start_time
        logging.info(f'FedNCA training completed in {elapsed:.2f}s')

    def _local_val_on_all_clients(self):
        for client_idx in range(self.client_num_in_total):
            if self.val_data_local_dict[client_idx] is None:
                continue
            client = self.client_list[client_idx]
            client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx], self.val_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx], self.train_data_local_num_dict[client_idx])
            metrics = client.local_validate()
            self.local_val_by_global_model[f'idx{client_idx}'].append(copy.deepcopy(metrics['test_acc']))
            logging.info(f"FedNCA val client {client_idx}: acc={metrics['test_acc']:.4f}, loss={metrics['test_loss']:.4f}")

    def _local_test_on_all_clients(self):
        for client_idx in range(self.client_num_in_total):
            if self.test_data_local_dict[client_idx] is None:
                continue
            client = self.client_list[client_idx]
            client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx], self.val_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx], self.train_data_local_num_dict[client_idx])
            metrics = client.local_test(True)
            self.local_performance_by_global_model[f'idx{client_idx}'].append(copy.deepcopy(metrics['test_acc']))
            logging.info(f"FedNCA test client {client_idx}: acc={metrics['test_acc']:.4f}, loss={metrics['test_loss']:.4f}")
