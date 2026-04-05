import copy
import logging
import os
import time
from collections import OrderedDict

import numpy as np
import torch

from .clientpfedfda import ClientpFedFDA


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


def interpolate_states(base_state, other_state, alpha):
    alpha = float(alpha)
    out = clone_state_dict(base_state)
    for k in out.keys():
        out[k] = (1.0 - alpha) * base_state[k].detach().cpu() + alpha * other_state[k].detach().cpu()
    return out


class pFedFDAAPI(object):
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
        self.client_stats = {}
        self.personalized_states = {}
        for client_idx in range(self.client_num_in_total):
            self.client_list.append(ClientpFedFDA(client_idx, train_data_local_dict[client_idx], val_data_local_dict[client_idx],
                                                  test_data_local_dict[client_idx], train_data_local_num_dict[client_idx],
                                                  args, device, model_trainer))

    def _client_sampling(self, round_idx):
        if self.client_num_in_total == self.client_num_per_round:
            return [cid for cid in range(self.client_num_in_total)]
        np.random.seed(round_idx)
        return np.random.choice(range(self.client_num_in_total), self.client_num_per_round, replace=False)

    def train(self):
        w_global = self.model_trainer.get_model_params()
        gamma = getattr(self.args, 'pfedfda_gamma', 0.5)
        start = time.time()
        for round_idx in range(self.args.comm_round):
            sampled = self._client_sampling(round_idx)
            local_results = {}
            for idx, client in enumerate(self.client_list[:len(sampled)]):
                cid = sampled[idx]
                client.update_local_dataset(cid, self.train_data_local_dict[cid], self.val_data_local_dict[cid],
                                            self.test_data_local_dict[cid], self.train_data_local_num_dict[cid])
                local_results[cid] = client.train(copy.deepcopy(w_global))
                self.client_stats[cid] = (local_results[cid]['logit_mean'], local_results[cid]['logit_std'])
            if not local_results:
                continue
            w_global = weighted_average([(self.train_data_local_num_dict[cid], local_results[cid]['state']) for cid in local_results])
            means = torch.stack([m for m, _ in self.client_stats.values()], dim=0)
            stds = torch.stack([s for _, s in self.client_stats.values()], dim=0)
            global_mean, global_std = means.mean(dim=0), stds.mean(dim=0)
            for cid, res in local_results.items():
                mean, std = res['logit_mean'], res['logit_std']
                dist = (mean - global_mean).pow(2).mean().sqrt() + (std - global_std).pow(2).mean().sqrt()
                alpha = float(torch.exp(-gamma * dist).clamp(0.1, 0.9))
                self.personalized_states[cid] = interpolate_states(w_global, res['state'], alpha)
            self.model_trainer.set_model_params(w_global)
            torch.save(w_global, os.path.join(self.args.save_path, f'{self.args.mode}_global_round{round_idx}'))
        logging.info(f'pFedFDA training completed in {time.time() - start:.2f}s')
