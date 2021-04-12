import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, stride=1, padding=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.convnet(x)


class ContractingPath(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convnet = DoubleConv(in_channels, out_channels)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.convnet(x)
        x = self.downsample(x)
        return x


class ExpansivePath(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        self.upconv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=2)
        self.convnet = DoubleConv(in_channels, out_channels, mid_channels=mid_channels)

    def forward(self, cache, x):
        x = self.upconv(x)
        diffY = cache.size()[2] - x.size()[2]
        diffX = cache.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([cache, x], dim=1)
        x = self.convnet(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.caches = []

        self.input_conv = DoubleConv(in_channels, 2 ** 6, stride=1, padding=1)
        self.left = nn.ModuleList([ContractingPath(2 ** i, 2 ** (i+1)) for i in range(6, 10)])
        self.right = nn.ModuleList([ExpansivePath(2 ** i, 2 ** (i-1)) for i in range(10, 6, -1)])
        self.conv1x1 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=1)

    def forward(self, x):
        x = self.input_conv(x)
        self.caches.append(x)
        for block in self.left:
            x = block(x)
            self.caches.append(x)
        del self.caches[-1]
        self.caches.reverse()
        for block, cache in zip(self.right, self.caches):
            x = block(cache, x)
        x = self.conv1x1(x)
        return x


if __name__ == "__main__":
    model = UNet()
    print(model)