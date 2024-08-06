import torch
import torch.nn as nn
import torch.optim as optim
# from torchvision import transforms
from torch.autograd import Variable
import numpy as np
import cv2
import os

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x, inplace=True)

        x = self.conv2(x)
        x = self.bn2(x)

        x += identity
        x = nn.functional.relu(x, inplace=True)

        return x


class ResNet(nn.Module):
    def __init__(self, num_blocks=[3, 4, 6, 3], num_classes=1):
        super(ResNet, self).__init__()

        self.in_channels = 16

        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(16, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(32, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(64, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(128, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResNetBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.pool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    # Load the ResNet model
    # Default is ResNet-18
    model = ResNet()

    # Load the pre-trained weights
    # model.load_state_dict(torch.load('resnet18.pth'))

    # Set the model to evaluate mode
    model.eval()

    # Example of how to use the model for prediction
    input_tensor = torch.randn(1, 1, 240, 320)
    output = model(input_tensor)

    # Print the prediction
    print(output)