import copy
import logging
import os
import time
from collections import OrderedDict

import numpy as np
import torch

from .clientgcml import ClientGCML


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


def flatten_state_dict(state, param_names=None):
    names = list(param_names) if param_names is not None else list(state.keys())
    flat = [state[k].detach().float().view(-1).cpu() for k in names]
    return torch.cat(flat, dim=0) if flat else torch.empty(0)


def cosine_similarity_from_states(state_a, state_b, param_names=None):
    a = flatten_state_dict(state_a, param_names)
    b = flatten_state_dict(state_b, param_names)
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    denom = torch.norm(a) * torch.norm(b)
    if denom.item() == 0:
        return 0.0
    return float(torch.dot(a, b) / denom)


def topk_similar_clients(similarity_row, self_idx, topk):
    cand = [(idx, sim) for idx, sim in enumerate(similarity_row.tolist()) if idx != self_idx]
    cand.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in cand[:topk]]


def interpolate_states(base_state, other_state, alpha):
    alpha = float(alpha)
    out = clone_state_dict(base_state)
    for k in out.keys():
        out[k] = (1.0 - alpha) * base_state[k].detach().cpu() + alpha * other_state[k].detach().cpu()
    return out


class GCMLAPI(object):
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
        self.personalized_states = {}
        for client_idx in range(self.client_num_in_total):
            self.client_list.append(ClientGCML(client_idx, train_data_local_dict[client_idx], val_data_local_dict[client_idx],
                                               test_data_local_dict[client_idx], train_data_local_num_dict[client_idx],
                                               args, device, model_trainer))

    def _client_sampling(self, round_idx):
        if self.client_num_in_total == self.client_num_per_round:
            return [cid for cid in range(self.client_num_in_total)]
        np.random.seed(round_idx)
        return np.random.choice(range(self.client_num_in_total), self.client_num_per_round, replace=False)

    def train(self):
        start = time.time()
        w_global = self.model_trainer.get_model_params()
        k = getattr(self.args, 'gcml_topk', 2)
        blend = getattr(self.args, 'gcml_blend', 0.5)
        for round_idx in range(self.args.comm_round):
            sampled = self._client_sampling(round_idx)
            local_states = {}
            for idx, client in enumerate(self.client_list[:len(sampled)]):
                cid = sampled[idx]
                client.update_local_dataset(cid, self.train_data_local_dict[cid], self.val_data_local_dict[cid],
                                            self.test_data_local_dict[cid], self.train_data_local_num_dict[cid])
                init_state = self.personalized_states.get(cid, w_global)
                local_states[cid] = client.train(copy.deepcopy(init_state))
            if not local_states:
                continue
            state_items = list(local_states.items())
            mat = np.zeros((len(state_items), len(state_items)), dtype=np.float32)
            for i, (_, si) in enumerate(state_items):
                for j, (_, sj) in enumerate(state_items):
                    mat[i, j] = cosine_similarity_from_states(si, sj)
            for i, (cid, state) in enumerate(state_items):
                nbr_ids = topk_similar_clients(mat[i], i, k)
                if not nbr_ids:
                    self.personalized_states[cid] = copy.deepcopy(state)
                    continue
                mixed = copy.deepcopy(state)
                for name in mixed.keys():
                    mixed[name] = torch.zeros_like(mixed[name])
                denom = 0.0
                for j in nbr_ids:
                    _, nbr_state = state_items[j]
                    weight = max(float(mat[i, j]), 0.0) + 1e-6
                    denom += weight
                    for name in mixed.keys():
                        mixed[name] += nbr_state[name] * weight
                for name in mixed.keys():
                    mixed[name] /= denom
                self.personalized_states[cid] = interpolate_states(state, mixed, blend)
            w_global = weighted_average([(self.train_data_local_num_dict[cid], local_states[cid]) for cid in local_states])
            self.model_trainer.set_model_params(w_global)
            torch.save(w_global, os.path.join(self.args.save_path, f'{self.args.mode}_global_round{round_idx}'))
        logging.info(f'GCML training completed in {time.time() - start:.2f}s')
