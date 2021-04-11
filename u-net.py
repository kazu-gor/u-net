import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def conv3x3(in_channels, out_channels, stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding)

def upconv(in_channels):
    return nn.Conv2d(in_channels, in_channels/2, kernel_size=2)

class LeftSideNet(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv =  conv3x3(num_channels, num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.downsample(x)
        return x

class RightSideNet(nn.Module):
    def __init__(self, num_channels):
        super(LeftSideNet, self).__init__()
        pass

    def forward(self, x):
        pass

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.leftconv = nn.ModuleList([LeftSideNet(2 ** i) for i in range(6, 11)])
        self.upconv()

    def forward(self, x):
        x = self.leftconv(x)
        

