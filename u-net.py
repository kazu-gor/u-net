import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchviz import make_dot


def left_side_first_block(in_channels):
    return nn.ModuleList([nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                          nn.ReLU(inplace=True),
                          nn.MaxPool2d(kernel_size=2, stride=2),
                          ])

def conv3x3_relu(num_channels, stride=1, padding=1):
    return nn.ModuleList([nn.Conv2d(num_channels, num_channels,
                                    kernel_size=3, stride=stride, padding=padding),
                          nn.ReLU(inplace=True),
                          ])

def upconv(num_channels):
    return nn.Conv2d(num_channels, num_channels // 2, kernel_size=2)


class LeftSideNet(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.convnet1 = conv3x3_relu(num_channels)
        self.convnet2 = conv3x3_relu(num_channels)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        for block in self.convnet1:
            x = block(x)
        for block in self.convnet2:
            x = block(x)
        cache = x
        x = self.downsample(x)
        return x, cache


class RightSideNet(nn.Module):
    def __init__(self, num_channels, index):
        super().__init__()
        self.upconv = upconv(num_channels)
        self.convnet1 = conv3x3_relu(num_channels // 2)
        self.convnet2 = conv3x3_relu(num_channels // 2)

    def forward(self, x, cache):
        x = upconv(x)
        x = torch.cat([cache, x])
        for block in self.convnet1:
            x = block(x)
        for block in self.convnet2:
            x = block(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.caches = []

        self.left_side_first_block = left_side_first_block(in_channels)
        self.leftconv = nn.ModuleList([LeftSideNet(2 ** i) for i in range(7, 11)])
        self.rightconv = nn.ModuleList([RightSideNet(2 ** (10 - i), i) for i in range(4)])
        self.conv1x1 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=1)

    def forward(self, x):
        for block in self.left_side:
            x = block(x)
        for block in self.leftconv:
            x, cache = block(x)
            self.caches.append(cache)
        for block, cache in zip(self.rightconv, caches):
            x = block(x, cache)
        x = self.conv1x1(x)
        return x


if __name__ == "__main__":
    model = UNet(in_channels=1, num_classes=2)
    print(model)