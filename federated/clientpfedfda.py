import copy
import torch


class ClientpFedFDA(object):
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

    def _estimate_logit_stats(self):
        model = self.model_trainer.model
        model.to(self.device)
        model.eval()
        logits = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.local_training_data):
                x = batch[0].to(self.device)
                out = model(x)
                logits.append(out.detach().float().mean(dim=tuple(range(2, out.dim()))) if out.dim() > 2 else out.detach().float())
                if batch_idx >= 1:
                    break
        if not logits:
            return torch.zeros(2), torch.ones(2)
        z = torch.cat(logits, dim=0)
        return z.mean(dim=0).cpu(), z.std(dim=0, unbiased=False).cpu() + 1e-6

    def train(self, init_state):
        self.model_trainer.set_model_params(init_state)
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        local_state = self.model_trainer.get_model_params()
        mean, std = self._estimate_logit_stats()
        return {'state': copy.deepcopy(local_state), 'logit_mean': mean, 'logit_std': std}
