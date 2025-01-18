import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import logging
import torch
import os
import copy
import torch.nn as nn
from torch.utils.data import DataLoader
from .DPF import DPF


class Client:
    def __init__(self, client_idx, local_training_data, local_val_data, local_test_data, local_sample_number, args, device, model_trainer):
        
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_val_data = local_val_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        self.DPF_times = args.DPF_times  

        self.model_trainer = model_trainer
        
        self.model = copy.deepcopy(self.model_trainer.model)

        print(f"Client {self.client_idx}: Model has been copied successfully.")

        self.eta = args.eta
        self.rand_percent = args.rand_percent
        self.layer_idx = args.layer_idx
        self.device = device
        
        if self.client_idx != -1: 
            if isinstance(self.local_training_data, DataLoader) and len(self.local_training_data) > 0:
                print(f"Client {self.client_idx}: Initializing DPF with DataLoader...")
                self.DPF = DPF(self.local_training_data, self.rand_percent, self.DPF_times, self.client_idx, self.device, self.layer_idx, self.eta)
            else:
                print(f"Client {self.client_idx}: No training data available, skipping DPF initialization.")
        else:
            print(f"Client {self.client_idx}: This is an OOD client. DPF initialization is not needed.")


        self.trajectory = None
        self.prev_weight = None

    def update_local_dataset(self, client_idx, local_training_data, local_val_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.model_trainer.set_id(client_idx)
        self.local_training_data = local_training_data
        self.local_val_data = local_val_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number

    def train(self):

        self.model_trainer.train(self.local_training_data, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        self.calcuate_trajectory(weights)
        self.prev_weight = weights
        return weights

    def local_validate(self, local_param=None):
        if local_param is not None:
            self.model_trainer.set_model_params(local_param)
        metrics = self.model_trainer.test(self.local_val_data, self.device, self.args)
        return metrics

    def local_test(self, b_use_test_dataset, local_param=None):
        if local_param is not None:
            self.model_trainer.set_model_params(local_param)

        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics

    def ood_test(self, ood_data, w_global):
        self.model_trainer.set_model_params(w_global)
        metrics = self.model_trainer.test(ood_data, self.device, self.args)
        return metrics

    def local_test_by_trajectory(self):
        model_trainer_copy = copy.deepcopy(self.model_trainer)
        model_trainer_copy.set_model_params(self.trajectory)
        metrics = model_trainer_copy.test(self.local_test_data, self.device, self.args)
        del model_trainer_copy
        return metrics

    def local_validate_by_trajectory(self):
        model_trainer_copy = copy.deepcopy(self.model_trainer)
        model_trainer_copy.set_model_params(self.trajectory)
        metrics = model_trainer_copy.test(self.local_val_data, self.device, self.args)
        del model_trainer_copy
        return metrics

    def ood_test_by_trajectory(self, ood_test_data):
        model_trainer_copy = copy.deepcopy(self.model_trainer)
        model_trainer_copy.set_model_params(self.trajectory)
        metrics = model_trainer_copy.test(ood_test_data, self.device, self.args)
        del model_trainer_copy
        return metrics

    # def save_trajectory(self, comm_round):
    #     torch.save(self.trajectory, os.path.join(self.args.save_path, "{}_idx_{}_round{}".format(self.args.mode, self.client_idx, comm_round)))
    def save_trajectory(self, comm_round):
        if comm_round == self.args.comm_round - 1:  # comm_round 是从0开始，最后一轮是 comm_round-1
            torch.save(
                self.model.state_dict(),
                os.path.join(self.args.save_path, "{}_idx_{}_round{}".format(self.args.mode, self.client_idx, comm_round))
            )

    def calcuate_trajectory(self, w_local):
        if self.trajectory == None:
            self.trajectory = w_local       
        else:
            for k in w_local.keys():
                self.trajectory[k] = self.args.alpha * self.trajectory[k] + (1-self.args.alpha) * w_local[k]

    def test_time_adaptation_by_iopfl(self, w_global):
        if w_global != None:
            self.model_trainer.set_model_params(w_global)
        metrics = self.model_trainer.io_pfl(self.local_test_data, self.device, self.args)
        return metrics
        
    def local_initialization(self, received_global_model):
        self.model.to(self.device) 
        self.ALA.adaptive_local_aggregation(received_global_model.to(self.device), self.model)  
