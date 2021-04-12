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

def double_conv(in_channels, out_channels, mid_channels=None, stride=1, padding=1):
    if not mid_channels:
        mid_channels = out_channels
    return nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, stride=stride, padding=padding),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, out_channels, stride=stride, padding=padding),
        nn.ReLU(inplace=True),
    )

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
        # x = self.downsample(x)
        return x

class ExpansivePath(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        self.upconv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=2)
        self.convnet = DoubleConv(in_channels, out_channels, mid_channels=mid_channels)

    def forward(self, cache, x):
        x = self.upconv(x)
        # print(f"x.shape: {x.shape}")
        # print(f"cache.shape: {cache.shape}")
        diffY = cache.size()[2] - x.size()[2]
        diffX = cache.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([cache, x], dim=1)
        # print(f"x.shape: {x.shape}")
        x = self.convnet(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
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
        print(f"len cache: {len(self.caches)}")
        for block, cache in zip(self.right, self.caches):
            print("*"*20)
            print(f"x.shape:    {x.shape}")
            print(f"cache.shape: {cache.shape}")
            x = block(cache, x)
        print(f"x.shape: {x.shape}")
        x = self.conv1x1(x)
        return x

# class LeftSideNet(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         # self.convnet1 = conv3x3_relu(num_channels)
#         self.convnet1 = DoubleConv(in_channels, out_channels)
#         # self.convnet2 = conv3x3_relu(num_channels)
#         self.convnet2 = DoubleConv(in_channels, out_channels)
#         self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

#     def forward(self, x):
#         for block in self.convnet1:
#             x = block(x)
#         for block in self.convnet2:
#             x = block(x)
#         cache = x
#         x = self.downsample(x)
#         return x, cache


# class RightSideNet(nn.Module):
#     def __init__(self, num_channels, index):
#         super().__init__()
#         self.upconv = upconv(num_channels)
#         self.convnet1 = conv3x3_relu(num_channels // 2)
#         self.convnet2 = conv3x3_relu(num_channels // 2)

#     def forward(self, x, cache):
#         x = upconv(x)
#         x = torch.cat([cache, x])
#         for block in self.convnet1:
#             x = block(x)
#         for block in self.convnet2:
#             x = block(x)
#         return x


# class UNet(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super().__init__()
#         self.caches = []

#         self.left_side_first_block = left_side_first_block(in_channels)
#         self.leftconv = nn.ModuleList([LeftSideNet(2 ** i) for i in range(7, 11)])
#         self.rightconv = nn.ModuleList([RightSideNet(2 ** (10 - i), i) for i in range(4)])
#         self.conv1x1 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=1)

#     def forward(self, x):
#         for block in self.left_side:
#             x = block(x)
#         for block in self.leftconv:
#             x, cache = block(x)
#             self.caches.append(cache)
#         for block, cache in zip(self.rightconv, caches):
#             x = block(x, cache)
#         x = self.conv1x1(x)
#         return x


if __name__ == "__main__":
    model = UNet(in_channels=224, num_classes=2)
    print(model)

    torch.manual_seed(0)
    datas = torch.randn(64, 224, 3, 3)
    y = model(datas)
    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.format = 'png'
    dot.render("graph_image")