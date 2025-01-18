# segmentation_models.py
import sys, os
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import BasicBlock, Bottleneck, resnet18, resnet34, resnet50
from torchvision.models._utils import IntermediateLayerGetter
from .routeconv import RouteConv2D, RouteConvTranspose2D

# 替换对 model_urls 的引用，手动设置 URLs
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
}

class ViewFlatten(nn.Module):
	def __init__(self):
		super(ViewFlatten, self).__init__()

	def forward(self, x):
		return x.view(x.size(0), -1)

# 定义 UNet 模型
class UNet(nn.Module):
    def __init__(self, input_shape, in_channels=3, out_channels=2, init_features=32, untrack_bn=False):
        super(UNet, self).__init__()
        if untrack_bn:
            bn_affine = False
            bn_track = False
        else:
            bn_affine = True
            bn_track = True

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1", bn_affine=bn_affine, bn_track=bn_track, prefix_name="encoder1.")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2", bn_affine=bn_affine, bn_track=bn_track, prefix_name="encoder2.")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3", bn_affine=bn_affine, bn_track=bn_track, prefix_name="encoder3.")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4", bn_affine=bn_affine, bn_track=bn_track, prefix_name="encoder4.")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck", bn_affine=bn_affine, bn_track=bn_track, prefix_name="bottleneck.")

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4", bn_affine=bn_affine, bn_track=bn_track, prefix_name="decoder4.")

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3", bn_affine=bn_affine, bn_track=bn_track, prefix_name="decoder3.")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2", bn_affine=bn_affine, bn_track=bn_track, prefix_name="decoder2.")

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet._block(features * 2, features, name="dec1", bn_affine=bn_affine, bn_track=bn_track, prefix_name="decoder1.")

        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)

    
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

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

        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, bn_affine, bn_track, name, prefix_name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "_conv1", nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False)),
                    (name + "_bn1", nn.BatchNorm2d(features)),
                    (name + "_relu1", nn.ReLU(inplace=True)),
                    (name + "_conv2", nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False)),
                    (name + "_bn2", nn.BatchNorm2d(features, affine=bn_affine, track_running_stats=bn_track)),
                    (name + "_relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

# 定义 ResnetUNet 模型
class ResnetUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, resnet_type='resnet50', pretrained=True):
        super(ResnetUNet, self).__init__()

        if resnet_type == 'resnet18':
            self.backbone = resnet18(pretrained=False)
            filters = [64, 64, 128, 256, 512]
        elif resnet_type == 'resnet34':
            self.backbone = resnet34(pretrained=False)
            filters = [64, 64, 128, 256, 512]
        elif resnet_type == 'resnet50':
            self.backbone = resnet50(pretrained=False)
            filters = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported ResNet type: {resnet_type}")

        # 加载预训练权重
        state_dict = torch.hub.load_state_dict_from_url(model_urls[resnet_type], progress=True)
        self.backbone.load_state_dict(state_dict)

        # Encoder layers (using ResNet's layers up to layer4 as encoder)
        self.encoder0 = nn.Sequential(self.backbone.conv1, self.backbone.bn1, self.backbone.relu)
        self.pool0 = self.backbone.maxpool
        self.encoder1 = self.backbone.layer1
        self.encoder2 = self.backbone.layer2
        self.encoder3 = self.backbone.layer3
        self.encoder4 = self.backbone.layer4

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(filters[4], filters[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[3]),
            nn.ReLU(inplace=True)
        )

        # Decoder layers
        self.upconv4 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.reduce_channels_4 = nn.Conv2d(filters[2] + filters[4], filters[2], kernel_size=1)
        self.decoder4 = self._block(filters[2], filters[2], name="dec4")

        self.upconv3 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.reduce_channels_3 = nn.Conv2d(filters[1] + filters[3], filters[1], kernel_size=1)
        self.decoder3 = self._block(filters[1], filters[1], name="dec3")

        self.upconv2 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.reduce_channels_2 = nn.Conv2d(filters[0] + filters[2], filters[0], kernel_size=1)
        self.decoder2 = self._block(filters[0], filters[0], name="dec2")

        self.upconv1 = nn.ConvTranspose2d(filters[0], filters[0], kernel_size=2, stride=2)
        self.reduce_channels_1 = nn.Conv2d(filters[0] + filters[1], filters[0], kernel_size=1)
        self.decoder1 = self._block(filters[0], filters[0], name="dec1")

        self.conv_final = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc0 = self.encoder0(x)
        enc1 = self.encoder1(self.pool0(enc0))
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)

        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = F.interpolate(dec4, size=enc4.shape[2:], mode='bilinear', align_corners=False)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.reduce_channels_4(dec4)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = F.interpolate(dec3, size=enc3.shape[2:], mode='bilinear', align_corners=False)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.reduce_channels_3(dec3)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = F.interpolate(dec2, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.reduce_channels_2(dec2)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = F.interpolate(dec1, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.reduce_channels_1(dec1)
        dec1 = self.decoder1(dec1)

        return self.conv_final(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "_conv1", nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False)),
                    (name + "_bn1", nn.BatchNorm2d(features)),
                    (name + "_relu1", nn.ReLU(inplace=True)),
                    (name + "_conv2", nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False)),
                    (name + "_bn2", nn.BatchNorm2d(features)),
                    (name + "_relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


