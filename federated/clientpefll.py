import copy

import torch


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


class ClientPeFLL(object):
    def __init__(self, client_idx, local_training_data, local_val_data, local_test_data,
                 local_sample_number, args, device, model_trainer):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_val_data = local_val_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.args = args
        self.device = device
        self.model_trainer = model_trainer

    def update_local_dataset(self, client_idx, local_training_data, local_val_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.model_trainer.set_id(client_idx)
        self.local_training_data = local_training_data
        self.local_val_data = local_val_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number

    def train(self, personalized_init):
        self.model_trainer.set_model_params(personalized_init)
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        local_state = self.model_trainer.get_model_params()
        descriptor = estimate_loader_descriptor(self.local_training_data, self.device)
        return {'personalized': copy.deepcopy(local_state), 'descriptor': descriptor}
