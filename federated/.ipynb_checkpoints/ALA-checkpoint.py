#怎么个自适应法
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import copy
import random
from torch.utils.data import DataLoader
import torch.nn.functional as F


import numpy as np

import torch
import torch.nn.functional as F

def custom_collate_fn(batch):
    # 获取批次中的最大图像大小
    max_height = max([b[0].shape[0] for b in batch])
    max_width = max([b[0].shape[1] for b in batch])
    max_channels = max([b[0].shape[2] for b in batch])

    # 获取批次中的最大标签大小
    max_label_batch = max([b[1].shape[0] for b in batch])
    max_label_channels = max([b[1].shape[1] for b in batch])
    max_label_height = max([b[1].shape[2] for b in batch])
    max_label_width = max([b[1].shape[3] for b in batch])

    padded_images = []
    padded_labels = []

    for image, label in batch:
        # 对图像进行填充，使得所有图像的大小一致
        pad_height = max_height - image.shape[0]
        pad_width = max_width - image.shape[1]
        pad_channels = max_channels - image.shape[2]
        padded_image = F.pad(image, (0, pad_width, 0, pad_height, 0,pad_channels), "constant", 0)
        padded_images.append(padded_image)

        # 对标签进行填充，使得所有标签的大小一致
        label_pad_batch= max_label_batch- label.shape[0]
        label_pad_channels = max_label_channels- label.shape[1]
        label_pad_height = max_label_height - label.shape[2]
        label_pad_width = max_label_width - label.shape[3]
        padded_label = F.pad(label, (0, label_pad_width, 0, label_pad_height,0, label_pad_channels, 0, label_pad_batch), "constant", 0)
        padded_labels.append(padded_label)

    # 将所有填充后的图像和标签合并
    return torch.stack(padded_images, 0), torch.stack(padded_labels, 0)


def crop_and_pad(image, target_size=(384, 384)):
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)

    print(f"Original image shape: {image.shape}")

    non_zero_coords = np.array(np.nonzero(image))
    if non_zero_coords.size == 0:
        print("Warning: Image is completely zero.")
        return np.zeros((target_size[0], target_size[1], image.shape[-1]), dtype=image.dtype)

    min_coords = non_zero_coords.min(axis=1)
    max_coords = non_zero_coords.max(axis=1)

    cropped_image = image[min_coords[0]:max_coords[0] + 1, min_coords[1]:max_coords[1] + 1, :]
    cropped_shape = cropped_image.shape
    pad_before = []
    pad_after = []

    for i in range(2):
        total_pad = target_size[i] - cropped_shape[i]
        if total_pad > 0:
            pad_before.append(total_pad // 2)
            pad_after.append(total_pad - pad_before[-1])
        else:
            pad_before.append(0)
            pad_after.append(0)

    if cropped_image.ndim == 2:
        cropped_image = np.expand_dims(cropped_image, axis=-1)

    print(f"Cropped image shape: {cropped_image.shape}")

    if len(cropped_image.shape) == 3 and cropped_image.shape[-1] == target_size[-1]:
        pad_width = ((pad_before[0], pad_after[0]), (pad_before[1], pad_after[1]), (0, 0))
    else:
        print(f"Warning: Cropped image shape {cropped_image.shape} does not match expected target size {target_size}.")
        return np.zeros((target_size[0], target_size[1], target_size[2] if len(target_size) > 2 else 1), dtype=image.dtype)

    try:
        padded_image = np.pad(cropped_image, pad_width, mode='constant', constant_values=0)
    except ValueError as e:
        print(f"Error during padding: {e}")
        print(f"Pad width: {pad_width}, Cropped image shape: {cropped_image.shape}")
        return np.zeros((target_size[0], target_size[1], cropped_image.shape[-1]), dtype=image.dtype)

    return padded_image

class ALA:
    def __init__(self,
                 loss_function: nn.Module,
                 train_data_loader: List[Tuple],
                 rand_percent: int,
                 layer_idx: int = 0,
                 eta: float = 1.0,
                 device: str = None,
                 ala_times: Optional[int] = None,
                 threshold: float = 0.1,
                 num_pre_loss: int = 10) -> None:
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        self.loss_function = loss_function
        self.train_data_loader = train_data_loader
        self.ala_times = ala_times
        self.rand_percent = rand_percent
        self.layer_idx = layer_idx
        self.eta = eta
        self.threshold = threshold
        self.num_pre_loss = num_pre_loss
        self.device = device

        self.weights = None
        self.start_phase = True

        if self.train_data_loader is not None:
            self.inspect_data_loader(train_data_loader)
        else:
            print("Warning: The train_data_loader is None. Skipping inspection.")

    def inspect_data_loader(self, data_loader):
        for batch_idx, (images, labels) in enumerate(data_loader):
            print(f"Batch {batch_idx}:")
            print(f"  Images type: {type(images)}, shape: {images.shape if isinstance(images, torch.Tensor) else 'N/A'}")
            print(f"  Labels type: {type(labels)}, shape: {labels.shape if isinstance(labels, torch.Tensor) else 'N/A'}")

    def adaptive_local_aggregation(self, global_model: nn.Module, local_model: nn.Module,  task="normal") -> None:
        global_model.to(self.device)
        local_model.to(self.device)

        if self.train_data_loader is None or len(self.train_data_loader) == 0:
            print("Warning: No training data available for adaptive local aggregation. Skipping this step.")
            return

        sampled_data = []
        if not hasattr(self.train_data_loader, 'dataset'):
            print("Warning: train_data_loader is not properly set or empty.")
            return
        rand_num = int(self.rand_percent * len(self.train_data_loader.dataset) / 100)
        for i, data in enumerate(self.train_data_loader):
            if len(sampled_data) >= rand_num:
                break
            if random.random() < (self.rand_percent / 100):
                sampled_data.append(data)

        if not sampled_data:
            print("Warning: No sampled data found. Skipping adaptive local aggregation.")
            return

        processed_sampled_data = []
        for image, label in sampled_data:
            if isinstance(image, torch.Tensor):
                image = image.numpy()

            padded_image = crop_and_pad(image, target_size=(384, 384))
            padded_tensor = torch.from_numpy(padded_image).float()
            print(f"Padded image shape before adding to dataset: {padded_tensor.shape}")
            processed_sampled_data.append((padded_tensor, label))

        if not processed_sampled_data:
            print("Warning: No processed data available after cropping and padding. Skipping adaptive local aggregation.")
            return

        actual_ala_times = len(processed_sampled_data) if self.ala_times is None else min(self.ala_times, len(processed_sampled_data))

        # 打印每一个处理过的数据样本的形状
        for idx, (image, label) in enumerate(processed_sampled_data):
            print(f"Processed sample {idx}: Image shape = {image.shape}, Label shape = {label.shape}")

        rand_loader = DataLoader(processed_sampled_data, batch_size=actual_ala_times, drop_last=True, shuffle=False, collate_fn=custom_collate_fn)

        for batch_idx, (x, y) in enumerate(rand_loader):
            print(f"rand_loader Batch {batch_idx}: x.shape={x.shape}, y.shape={y.shape}")

            
        # obtain the references of the parameters
        params_g = list(global_model.parameters())
        params = list(local_model.parameters())

        # deactivate ALA at the 1st communication iteration
        if torch.sum(params_g[0] - params[0]) == 0:
            return

        # preserve all the updates in the lower layers
        for param, param_g in zip(params[:-self.layer_idx], params_g[:-self.layer_idx]):
            param.data = param_g.data.clone()


        # temp local model only for weight learning
        model_t = copy.deepcopy(local_model)
        model_t.to(self.device)  # 将模型参数转移到设备上
        params_t = list(model_t.parameters())

        # only consider higher layers
        params_p = params[-self.layer_idx:]
        params_gp = params_g[-self.layer_idx:]
        params_tp = params_t[-self.layer_idx:]

        # frozen the lower layers to reduce computational cost in Pytorch
        for param in params_t[:-self.layer_idx]:
            param.requires_grad = False

        # used to obtain the gradient of higher layers
        # no need to use optimizer.step(), so lr=0
        optimizer = torch.optim.SGD(params_tp, lr=0)

        # initialize the weight to all ones in the beginning
        if self.weights == None:
            self.weights = [torch.ones_like(param.data).to(self.device) for param in params_p]

        # initialize the higher layers in the temp local model
        for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp,
                                                self.weights):
            param_t.data = (param + (param_g - param) * weight).to(self.device)

        # weight learning
        losses = []  # record losses
        cnt = 0  # weight training iteration counter
        while True:
            for x, y in rand_loader:
                # 移动 x 和 y 到指定的设备上
                x = x.to(self.device)
                y = y.to(self.device)

                    
                optimizer.zero_grad()
                output = model_t(x, task)
                loss_value = self.loss(output, y) # modify according to the local objective
                loss_value.backward()

                # update weight in this batch
                for param_t, param, param_g, weight in zip(params_tp, params_p,
                                                        params_gp, self.weights):
                    weight.data = torch.clamp(
                        weight - self.eta * (param_t.grad * (param_g - param)), 0, 1).to(self.device)

                # update temp local model in this batch
                for param_t, param, param_g, weight in zip(params_tp, params_p,
                                                        params_gp, self.weights):
                    param_t.data = (param + (param_g - param) * weight).to(self.device)

            losses.append(loss_value.item())
            cnt += 1

            # only train one epoch in the subsequent iterations
            if not self.start_phase:
                break

            # train the weight until convergence
            if len(losses) > self.num_pre_loss and np.std(losses[-self.num_pre_loss:]) < self.threshold:
                print('Client:', self.cid, '\tStd:', np.std(losses[-self.num_pre_loss:]),
                    '\tALA epochs:', cnt)
                break

        self.start_phase = False

        # obtain initialized local model
        for param, param_t in zip(params_p, params_tp):
            param.data = param_t.data.clone()