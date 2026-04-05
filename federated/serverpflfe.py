import copy
import logging
import os
import time
from collections import OrderedDict

import numpy as np
import torch

from .clientpflfe import ClientpFLFE


def clone_state_dict(state_dict):
    return OrderedDict((k, v.detach().cpu().clone()) for k, v in state_dict.items())


def weighted_average(states):
    total = float(sum(n for n, _ in states))
    avg = clone_state_dict(states[0][1])
    for k in avg.keys():
        avg[k] = torch.zeros_like(avg[k])
    for n, st in states:
        w = n / total
        for k in avg.keys():
            avg[k] += st[k].detach().cpu() * w
    return avg


class pFLFEAPI(object):
    def __init__(self, dataset, device, args, model_trainer):
        self.device = device
        self.args = args
        client_num, [_, _, _, train_data_local_num_dict, train_data_local_dict,
                     val_data_local_dict, test_data_local_dict, ood_data] = dataset
        self.client_num_in_total = client_num
        self.client_num_per_round = int(self.client_num_in_total * self.args.percent)
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.val_data_local_dict = val_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.ood_data = ood_data
        self.model_trainer = model_trainer
        self.client_list = []
        for client_idx in range(self.client_num_in_total):
            self.client_list.append(ClientpFLFE(client_idx, train_data_local_dict[client_idx], val_data_local_dict[client_idx],
                                                test_data_local_dict[client_idx], train_data_local_num_dict[client_idx],
                                                args, device, model_trainer))

    def _client_sampling(self, round_idx):
        if self.client_num_in_total == self.client_num_per_round:
            return [cid for cid in range(self.client_num_in_total)]
        np.random.seed(round_idx)
        return np.random.choice(range(self.client_num_in_total), self.client_num_per_round, replace=False)

    def train(self):
        w_global = self.model_trainer.get_model_params()
        start = time.time()
        for round_idx in range(self.args.comm_round):
            sampled = self._client_sampling(round_idx)
            w_locals = []
            for idx, client in enumerate(self.client_list[:len(sampled)]):
                cid = sampled[idx]
                client.update_local_dataset(cid, self.train_data_local_dict[cid], self.val_data_local_dict[cid],
                                            self.test_data_local_dict[cid], self.train_data_local_num_dict[cid])
                local_state = client.train(copy.deepcopy(w_global))
                w_locals.append((client.get_sample_number(), copy.deepcopy(local_state)))
            w_global = weighted_average(w_locals)
            self.model_trainer.set_model_params(w_global)
            torch.save(w_global, os.path.join(self.args.save_path, f'{self.args.mode}_global_round{round_idx}'))
        logging.info(f'pFLFE training completed in {time.time() - start:.2f}s')
