import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet,Bottleneck

# 替换对 model_urls 的引用，手动设置 URLs
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    # 添加其他需要的模型 URL
}

# 定义一个自定义层，用于将输入展平（将多维张量展平成二维）
class ViewFlatten(nn.Module):
	def __init__(self):
		super(ViewFlatten, self).__init__()

	def forward(self, x):
		return x.view(x.size(0), -1)


# 定义U-Net模型类  这是一个基于 U-Net 架构的卷积神经网络模型，用于图像分割和自监督任务
class UNet(nn.Module):
    # __init__ 方法初始化模型各层，包括编码器、池化层、瓶颈层、解码器、转置卷积层、卷积层、批量归一化层、ReLU激活函数、平均池化层和全连接层
    def __init__(self, input_shape, in_channels=3, out_channels=2, init_features=32, untrack_bn=False):
        super(UNet, self).__init__()

        # 如果不跟踪BN层，则设置bn_affine和bn_track为False
        if untrack_bn:
            bn_affine = False
            bn_track = False
        else:
            bn_affine = True
            bn_track = True


        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1", bn_affine=bn_affine, bn_track=bn_track)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2", bn_affine=bn_affine, bn_track=bn_track)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3", bn_affine=bn_affine, bn_track=bn_track)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4", bn_affine=bn_affine, bn_track=bn_track)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck", bn_affine=bn_affine, bn_track=bn_track)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4", bn_affine=bn_affine, bn_track=bn_track)
        
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2,
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3", bn_affine=bn_affine, bn_track=bn_track)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2", bn_affine=bn_affine, bn_track=bn_track)
       
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2,
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1", bn_affine=bn_affine, bn_track=bn_track)

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(24)
        self.fc = nn.Linear(512, 4)
        self.flat = ViewFlatten()

      
    # forward 方法定义了前向传播过程，根据任务类型（分割或自监督）决定执行不同的操作
    def forward(self, x, task):

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        
        bottleneck = self.bottleneck(self.pool4(enc4))
        if task == "self":
            logits = self.fc(self.flat(self.avgpool(self.relu(self.bn(bottleneck)))))
            return logits

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        dec1 = self.conv(dec1)

        return dec1

    # _block 静态方法：
    # 用于创建模型中的基本块（卷积层、批量归一化层、ReLU激活函数），这些块在编码器和解码器中反复使用。
    @staticmethod
    def _block(in_channels, features, bn_affine, bn_track, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "_conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "_bn1", nn.BatchNorm2d(num_features=features)),
                    (name + "_relu1", nn.ReLU(inplace=True)),
                    (
                        name + "_conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "_bn2", nn.BatchNorm2d(num_features=features, affine=bn_affine, track_running_stats=bn_track)),
                    (name + "_relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

# 空函数，仅用于检查该模块是否作为主程序运行
if __name__=='__main__':
    pass