import copy
import torch


def get_arg(args, name, default):
    return getattr(args, name, default) if args is not None else default


class ClientpFLFE(object):
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

    def _feature_enhancement_stage(self):
        model = self.model_trainer.model
        model.to(self.device)
        model.train()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=get_arg(self.args, 'pflfe_fe_lr', self.args.lr))
        steps = get_arg(self.args, 'pflfe_fe_steps', 1)
        for _ in range(steps):
            for batch in self.local_training_data:
                x = batch[0].to(self.device)
                noise = torch.randn_like(x) * 0.05
                x_aug = torch.clamp(x + noise, x.min().item(), x.max().item())
                pred = model(x)
                pred_aug = model(x_aug)
                p = torch.softmax(pred, dim=1) if pred.dim() > 1 and pred.size(1) > 1 else torch.sigmoid(pred)
                q = torch.softmax(pred_aug, dim=1) if pred_aug.dim() > 1 and pred_aug.size(1) > 1 else torch.sigmoid(pred_aug)
                loss_cons = torch.mean((p - q) ** 2)
                if pred.dim() > 1 and pred.size(1) > 1:
                    entropy = -(p.clamp_min(1e-8) * p.clamp_min(1e-8).log()).sum(dim=1).mean()
                else:
                    entropy = -(p.clamp(1e-8, 1 - 1e-8) * p.clamp(1e-8, 1 - 1e-8).log() +
                                (1 - p).clamp(1e-8, 1 - 1e-8) * (1 - p).clamp(1e-8, 1 - 1e-8).log()).mean()
                loss = get_arg(self.args, 'pflfe_cons_w', 1.0) * loss_cons + get_arg(self.args, 'pflfe_ent_w', 0.05) * entropy
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                break

    def train(self, w_global):
        self.model_trainer.set_model_params(w_global)
        self._feature_enhancement_stage()
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        return copy.deepcopy(self.model_trainer.get_model_params())
