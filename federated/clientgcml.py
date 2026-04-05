import copy


class ClientGCML(object):
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

    def train(self, init_state):
        self.model_trainer.set_model_params(init_state)
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        return copy.deepcopy(self.model_trainer.get_model_params())
