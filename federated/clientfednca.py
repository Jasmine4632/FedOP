import copy
from collections import OrderedDict


def get_arg(args, name, default):
    return getattr(args, name, default) if args is not None else default


def select_param_names(state_dict, include_keywords=None, exclude_keywords=None):
    include_keywords = list(include_keywords or [])
    exclude_keywords = list(exclude_keywords or [])
    selected = []
    for name in state_dict.keys():
        include_ok = True if not include_keywords else any(key in name for key in include_keywords)
        exclude_ok = not any(key in name for key in exclude_keywords)
        if include_ok and exclude_ok:
            selected.append(name)
    return selected


class ClientFedNCA(object):
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

    def local_validate(self, local_param=None):
        if local_param is not None:
            self.model_trainer.set_model_params(local_param)
        return self.model_trainer.test(self.local_val_data, self.device, self.args)

    def local_test(self, b_use_test_dataset, local_param=None):
        if local_param is not None:
            self.model_trainer.set_model_params(local_param)
        test_data = self.local_test_data if b_use_test_dataset else self.local_training_data
        return self.model_trainer.test(test_data, self.device, self.args)

    def train(self, w_global):
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        local_state = self.model_trainer.get_model_params()

        keywords = get_arg(self.args, 'fednca_keywords', ['decoder', 'seg', 'head', 'classifier', 'outc'])
        selected = select_param_names(local_state, include_keywords=keywords)
        if not selected:
            selected = list(local_state.keys())

        partial_state = OrderedDict()
        for name in selected:
            tensor = local_state[name].detach().cpu().clone()
            denom = tensor.norm().item()
            if denom > 0:
                tensor = tensor / denom
            partial_state[name] = tensor

        return {
            'full': copy.deepcopy(local_state),
            'partial': partial_state,
            'selected_names': selected,
        }
