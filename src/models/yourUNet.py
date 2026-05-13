# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
Modified by: Mamadou Mountaha Ndoye
Modifications apportées :
    1. Blocs résiduels (ResNet-style)
    2. Convolutions séparables en profondeur (depthwise separable)
    3. Module ASPP au bottleneck
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.CNNBaseModel import CNNBaseModel


class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        # Combinaison des canaux avec une conv 1x1
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, use_sep_conv=False):
        super().__init__()

        if use_sep_conv:
            self.conv1 = DepthwiseSeparableConv(in_channels, out_channels)
            self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        else:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1, bias=False
            )
            self.conv2 = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=1, bias=False
            )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Conv1x1 + BN si canaux diffèrent

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, branch_channels=256):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                branch_channels,
                kernel_size=3,
                padding=3,
                dilation=3,
                bias=False,
            ),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                branch_channels,
                kernel_size=3,
                padding=6,
                dilation=6,
                bias=False,
            ),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )

        self.branch4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(
                branch_channels * 4, branch_channels * 4, kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(branch_channels * 4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        b4 = F.interpolate(
            self.branch4(x), size=x.shape[2:], mode="bilinear", align_corners=False
        )
        out = torch.cat([b1, b2, b3, b4], dim=1)
        return self.fusion(out)


class YourUNet(CNNBaseModel):

    def __init__(self, num_classes=4, init_weights=True):
        super().__init__(num_classes, init_weights)

        self.enc1 = ResidualBlock(1, 64, use_sep_conv=False)

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.enc2 = ResidualBlock(64, 128, use_sep_conv=True)
        self.enc3 = ResidualBlock(128, 256, use_sep_conv=True)
        self.enc4 = ResidualBlock(256, 512, use_sep_conv=True)

        self.aspp = ASPPModule(in_channels=512, branch_channels=256)
        self.bridge_up = nn.ConvTranspose2d(
            1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1
        )

        self.dec4 = ResidualBlock(512 + 512, 256, use_sep_conv=True)
        self.up4 = nn.ConvTranspose2d(
            256, 256, kernel_size=3, stride=2, padding=1, output_padding=1
        )

        self.dec3 = ResidualBlock(256 + 256, 128, use_sep_conv=True)
        self.up3 = nn.ConvTranspose2d(
            128, 128, kernel_size=3, stride=2, padding=1, output_padding=1
        )

        self.dec2 = ResidualBlock(128 + 128, 64, use_sep_conv=True)
        self.up2 = nn.ConvTranspose2d(
            64, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        )

        self.dec1 = ResidualBlock(64 + 64, 64, use_sep_conv=False)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        bridge = self.aspp(self.pool(e4))
        up = self.bridge_up(bridge)

        d4 = self.dec4(torch.cat([up, e4], dim=1))
        d4 = self.up4(d4)

        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d3 = self.up3(d3)

        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d2 = self.up2(d2)

        d1 = self.dec1(torch.cat([d2, e1], dim=1))

        return self.final_conv(d1)


"""
Fin de votre code.
"""
