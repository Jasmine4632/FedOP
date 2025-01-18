import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import h5py
import scipy.io as scio
from glob import glob
import SimpleITK as sitk
import random
import cv2



class Prostate(Dataset):
    def __init__(self, site, base_path=None, split='train', transform=None):
        '''
        :param site: 站点名称，如 'austin', 'chicago', 等
        :param base_path: 数据集的基础路径
        :param split: 数据集分割，可以是 'train', 'val', 或 'test'
        :param transform: 需要应用的数据预处理变换
        '''
        # 新的数据集名称列表
        self.sites = ['austin', 'chicago', 'kitsap', 'massachusetts', 'tyrol','vienna', 'whu']
        
        # 确保 site 在定义的站点列表中
        assert site in self.sites, f"Site must be one of {self.sites}"

        self.site = site
        self.split = split
        self.transform = transform
        
        # 默认的基础路径，如果没有提供则使用默认路径
        self.base_path = base_path if base_path is not None else '/root/autodl-tmp/pythonproject/IOP-FL-main/data/prostate/dataset'
        
        # 根据站点和数据划分生成文件路径
        image_dir = os.path.join(self.base_path, site, 'images')
        label_dir = os.path.join(self.base_path, site, 'labels')
        
        # 确保文件夹存在
        assert os.path.exists(image_dir), f"Image folder does not exist at {image_dir}"
        assert os.path.exists(label_dir), f"Label folder does not exist at {label_dir}"
        
        # 获取所有图像文件
        image_files = sorted(os.listdir(image_dir))
        label_files = sorted(os.listdir(label_dir))
        
        # 确保图像和标签文件数量匹配
        assert len(image_files) == len(label_files), "Mismatch between image and label counts"
        
        # 打乱图像和标签文件
        combined = list(zip(image_files, label_files))
        random.shuffle(combined)  # 随机打乱数据
        image_files, label_files = zip(*combined)  # 解压打乱后的图像和标签文件
        
        # 根据 split 参数选择训练、验证或测试数据
        num_images = len(image_files)
        if self.split == 'train':
            self.image_files = image_files[:int(num_images * 0.7)]  # 70% for training
            self.label_files = label_files[:int(num_images * 0.7)]
        elif self.split == 'val':
            self.image_files = image_files[int(num_images * 0.7):int(num_images * 0.85)]  # 15% for validation
            self.label_files = label_files[int(num_images * 0.7):int(num_images * 0.85)]
        elif self.split == 'test':
            self.image_files = image_files[int(num_images * 0.85):]  # 15% for testing
            self.label_files = label_files[int(num_images * 0.85):]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 获取图像和标签路径
        image_path = os.path.join(self.base_path, self.site, 'images', self.image_files[idx])
        label_path = os.path.join(self.base_path, self.site, 'labels', self.label_files[idx])


        # 加载图像
        image = Image.open(image_path).convert('RGB')  # 使用 PIL 打开图像并确保是 RGB 模式
        image = np.array(image)  # 将图像转换为 numpy 数组

        # 加载标签（标签是 .png 格式）
        label = Image.open(label_path).convert('L')  # 标签是单通道（L）图像
        label = np.array(label)  # 将标签转换为 numpy 数组

        # 确保标签是二值化的（根据需要，可以添加标签的处理）
        label = np.where(label > 0, 1, 0)  # 假设标签是分割图像，0 是背景，1 是前景

        # 数据增强操作（如果需要）
        if self.transform:
            image, label = self.transform(image, label)

        # 转换为 tensor
        image = np.transpose(image, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        image = torch.Tensor(image)
        label = torch.Tensor(label)[None, :]  # 给标签添加一个维度 (H, W) -> (1, H, W)

        return image, label



# 这个块在脚本作为主程序执行时运行。这里的 pass 表示什么都不做，这个块可以用于测试或调试目的
if __name__=='__main__':
    pass


