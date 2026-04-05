import copy
import logging
import os
import time
from collections import OrderedDict

import numpy as np
import torch

from .clienthyperfedzero import ClientHyperFedZero


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


def state_difference(local_state, global_state, param_names=None):
    names = list(param_names) if param_names is not None else list(local_state.keys())
    diff = OrderedDict()
    for k in names:
        diff[k] = local_state[k].detach().cpu() - global_state[k].detach().cpu()
    return diff


def unflatten_like(flat_tensor, reference_state, names):
    out = OrderedDict()
    offset = 0
    for name in names:
        ref = reference_state[name]
        numel = ref.numel()
        out[name] = flat_tensor[offset: offset + numel].view_as(ref).detach().cpu().clone()
        offset += numel
    return out


def estimate_loader_descriptor(loader, device, max_batches=2):
    means, stds = [], []
    if loader is None:
        return torch.zeros(6)
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device).float()
        dims = tuple(range(2, x.dim())) if x.dim() > 2 else (0,)
        means.append(x.mean(dim=dims).mean(dim=0).detach().cpu())
        stds.append(x.std(dim=dims, unbiased=False).mean(dim=0).detach().cpu())
    if not means:
        return torch.zeros(6)
    mean = torch.stack(means, dim=0).mean(dim=0).flatten()
    std = torch.stack(stds, dim=0).mean(dim=0).flatten()
    desc = torch.cat([mean[:3], std[:3]], dim=0)
    if desc.numel() < 6:
        desc = torch.nn.functional.pad(desc, (0, 6 - desc.numel()))
    return desc[:6].float()


class SmallHyperNetwork(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def fit_hypernetwork(descriptors, targets, hidden_dim=64, lr=1e-3, steps=200, device='cpu'):
    model = SmallHyperNetwork(descriptors.size(1), targets.size(1), hidden_dim=hidden_dim).to(device)
    descriptors = descriptors.to(device)
    targets = targets.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        pred = model(descriptors)
        loss = loss_fn(pred, targets)
        loss.backward()
        optimizer.step()
    model.eval()
    return model.cpu()


class HyperFedZeroAPI(object):
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
        self.hypernet = None
        self.target_names = None
        for client_idx in range(self.client_num_in_total):
            self.client_list.append(ClientHyperFedZero(client_idx, train_data_local_dict[client_idx], val_data_local_dict[client_idx],
                                                       test_data_local_dict[client_idx], train_data_local_num_dict[client_idx],
                                                       args, device, model_trainer))

    def _client_sampling(self, round_idx):
        if self.client_num_in_total == self.client_num_per_round:
            return [cid for cid in range(self.client_num_in_total)]
        np.random.seed(round_idx)
        return np.random.choice(range(self.client_num_in_total), self.client_num_per_round, replace=False)

    def _get_target_names(self, state_dict):
        if self.target_names is None:
            names = [k for k in state_dict.keys() if any(t in k for t in ['decoder', 'seg', 'head', 'classifier', 'outc'])]
            self.target_names = names if names else list(state_dict.keys())[-4:]
        return self.target_names

    def train(self):
        w_global = self.model_trainer.get_model_params()
        start = time.time()
        for round_idx in range(self.args.comm_round):
            sampled = self._client_sampling(round_idx)
            local_results = {}
            descs, deltas = [], []
            names = self._get_target_names(w_global)
            for idx, client in enumerate(self.client_list[:len(sampled)]):
                cid = sampled[idx]
                client.update_local_dataset(cid, self.train_data_local_dict[cid], self.val_data_local_dict[cid],
                                            self.test_data_local_dict[cid], self.train_data_local_num_dict[cid])
                local_results[cid] = client.train(copy.deepcopy(w_global))
                descs.append(local_results[cid]['descriptor'])
                delta = state_difference(local_results[cid]['state'], w_global, names)
                deltas.append(flatten_state_dict(delta, names))
            if local_results:
                w_global = weighted_average([(self.train_data_local_num_dict[cid], res['state']) for cid, res in local_results.items()])
                self.model_trainer.set_model_params(w_global)
                self.hypernet = fit_hypernetwork(torch.stack(descs, dim=0), torch.stack(deltas, dim=0),
                                                 hidden_dim=getattr(self.args, 'hyperfedzero_hidden_dim', 64),
                                                 lr=getattr(self.args, 'hyperfedzero_hn_lr', 1e-3),
                                                 steps=getattr(self.args, 'hyperfedzero_hn_steps', 200),
                                                 device=str(self.device))
                torch.save(w_global, os.path.join(self.args.save_path, f'{self.args.mode}_global_round{round_idx}'))
        logging.info(f'HyperFedZero training completed in {time.time() - start:.2f}s')

    def generate_model_for_loader(self, loader):
        global_state = self.model_trainer.get_model_params()
        if self.hypernet is None:
            return clone_state_dict(global_state)
        descriptor = estimate_loader_descriptor(loader, self.device)
        names = self._get_target_names(global_state)
        with torch.no_grad():
            delta_vec = self.hypernet(descriptor.view(1, -1)).squeeze(0)
        delta_state = unflatten_like(delta_vec, global_state, names)
        generated = clone_state_dict(global_state)
        for name in names:
            generated[name] = generated[name] + delta_state[name]
        return generated

    def generate_ood_model(self):
        return self.generate_model_for_loader(self.ood_data)
