# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License:
Other: Suggestions are welcome
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.CNNBaseModel import CNNBaseModel


class YourSegNet(CNNBaseModel):
    """
    Non-symmetric encoder-decoder architecture for cardiac MRI segmentation.
    Based on DeepLabV3-like design with dilated convolutions and multi-scale context.
    """

    def __init__(self, num_classes=4, init_weights=True):
        """
        Builds YourSegNet  model.
        Args:
            num_classes(int): number of classes. default 10(cifar10 or svhn)
            init_weights(bool): when true uses _initialize_weights function to initialize
            network's weights.
        """

        super().__init__(num_classes, init_weights=False)
        
        # Encoder: aggressive downsampling (non-symmetric)
        self.stem = ConvBNReLU(1, 32, kernel_size=3, stride=2)  # 1/2
        self.enc1 = ResidualBlock(32, 64, stride=2)              # 1/4
        self.enc2 = ResidualBlock(64, 128, stride=2)             # 1/8
        self.enc3 = ResidualBlock(128, 256, stride=2)            # 1/16
        self.detail_fuse = ConvBNReLU(32, 32, kernel_size=1, stride=1)
        
        # Multi-scale context at bottleneck (ASPP-like)
        self.aspp = ASPP(256, 256)
        
        # Non-symmetric decoder: lightweight upsampling
        self.dec1 = ASPPBlock(256, 128)  # 1/8
        self.dec2 = ASPPBlock(128, 64)   # 1/4
        self.dec3 = ASPPBlock(64, 32)    # 1/2
        
        # Output head
        self.final_conv = nn.Sequential(
            ConvBNReLU(32, 32, kernel_size=3),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        input_size = x.shape[-2:]
        
        # Encoder
        s0 = self.stem(x)       # 1/2
        e1 = self.enc1(s0)      # 1/4
        e2 = self.enc2(e1)      # 1/8
        e3 = self.enc3(e2)      # 1/16
        
        # Bottleneck context
        c = self.aspp(e3)       # 1/16
        
        # Non-symmetric decoder (no dense skip connections like UNet)
        d1 = self.dec1(c)       # 1/8
        d2 = self.dec2(d1)      # 1/4
        d3 = self.dec3(d2)      # 1/2

        detail = self.detail_fuse(s0)
        detail = F.interpolate(detail, size=d3.shape[-2:], mode='bilinear', align_corners=False)
        d3 = d3 + detail
        
        # Final output
        out = self.final_conv(d3)  # 1/1
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        
        return out


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=stride)
        self.conv2 = ConvBNReLU(out_channels, out_channels, kernel_size=3, stride=1)
        
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        return self.relu(out)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling for multi-scale context"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvBNReLU(in_channels, out_channels, kernel_size=1)
        )
        
        self.aspp1 = ConvBNReLU(in_channels, out_channels, kernel_size=1)
        self.aspp2 = ConvBNReLU(in_channels, out_channels, kernel_size=3, dilation=6)
        self.aspp3 = ConvBNReLU(in_channels, out_channels, kernel_size=3, dilation=12)
        self.aspp4 = ConvBNReLU(in_channels, out_channels, kernel_size=3, dilation=18)
        
        self.project = nn.Sequential(
            ConvBNReLU(out_channels * 5, out_channels, kernel_size=1),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        size = x.shape[-2:]
        
        pool = self.global_pool(x)
        pool = F.interpolate(pool, size=size, mode='bilinear', align_corners=False)
        
        aspp1 = self.aspp1(x)
        aspp2 = self.aspp2(x)
        aspp3 = self.aspp3(x)
        aspp4 = self.aspp4(x)
        
        out = torch.cat([pool, aspp1, aspp2, aspp3, aspp4], dim=1)
        return self.project(out)


class ASPPBlock(nn.Module):
    """Decoder block with upsampling and dilated conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, kernel_size=3, dilation=1),
            ConvBNReLU(out_channels, out_channels, kernel_size=3, dilation=1)
        )
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.conv(x)