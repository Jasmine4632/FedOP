from abc import ABC, abstractmethod
import os
import torch

class ModelTrainer(ABC):
    """Abstract base class for federated learning trainer."""

    def __init__(self, model, args=None):
        self.model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  # Move model to device immediately
        self.args = args
        self.id = 0  # Initialize ID to 0
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []
        self.test_loss = []
        self.test_acc = []
        
        # Check if save_path attribute exists in args and handle it appropriately
        if self.args and hasattr(self.args, 'save_path'):
            if not os.path.exists(self.args.save_path):
                os.makedirs(self.args.save_path)  # Create directory if it does not exist
        else:
            print("Warning: 'save_path' not found in args. Using default path.")
            self.args.save_path = './default_save_path'  # Set a default value

    def set_id(self, trainer_id):
        self.id = trainer_id

    @abstractmethod
    def get_model_params(self, model_type="global"):
        """
        Abstract method to get model parameters.

        Args:
            model_type (str): 'global' or 'personal'. Default is 'global'.
        """
        pass

    @abstractmethod
    def set_model_params(self, model_parameters, model_type="global"):
        """
        Abstract method to set model parameters.

        Args:
            model_parameters: The parameters to set.
            model_type (str): 'global' or 'personal'. Default is 'global'.
        """
        pass
