### based on https://github.com/kuangliu/pytorch-cifar

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, self.expansion*out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(self.expansion*out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*out_channels)
            )
        self.act = nn.ReLU()

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)      
        x = self.conv3(x)
        x = self.bn3(x)
        x += self.shortcut(input)
        x = self.act(x)
        return x
    

class ResNet(nn.Module):
    def __init__(self, block_depths, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(Bottleneck, 64, block_depths[0], stride=1)
        self.layer2 = self._make_layer(Bottleneck, 128, block_depths[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, block_depths[2], stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, block_depths[3], stride=2)
        self.linear = nn.Linear(512*Bottleneck.expansion, num_classes)

    def _make_layer(self, block, num_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        print(strides)
        for stride in strides:
            layers.append(block(self.in_channels, num_channels, stride))
            self.in_channels = num_channels * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


