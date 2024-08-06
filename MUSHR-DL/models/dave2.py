import torch
import torch.nn as nn
import torch.optim as optim
# from torchvision import transforms
from torch.autograd import Variable
import numpy as np
import cv2
import os

class NVIDIA_Dave2(nn.Module):

    def __init__(self):
        super(NVIDIA_Dave2, self).__init__()

        self.conv1 = nn.Conv2d(1, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(48576, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):

        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.conv4(x)
        x = nn.functional.relu(x)
        x = self.conv5(x)
        x = nn.functional.relu(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)

        return x


if __name__ == '__main__':
    # Load the NVIDIA Dave2 model
    model = NVIDIA_Dave2()

    # Load the pre-trained weights
    # model.load_state_dict(torch.load('nvidia_dave2_weights.pth'))

    # Set the model to evaluate mode
    model.eval()

    # Example of how to use the model for prediction
    input_tensor = torch.randn(1, 1, 240, 320)
    output = model(input_tensor)

    # Print the prediction
    print(output)