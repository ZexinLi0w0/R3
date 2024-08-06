import numpy as np
import torch
import torch.nn.functional as F

# Load the dataset
dataset = np.load('MUSHR_320x240_training.npy',allow_pickle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Extract images and labels from the dataset
'''
    explain to data:
    data[:][0] is image
    data[:][3] is position
        data[:][3][3] is speed
        data[:][3][4] is steering
'''

X = torch.Tensor([i[0] for i in dataset]).unsqueeze(1).to(device) # [n, 1, 90, 160]
X = F.interpolate(X, size=(240, 320), mode='bilinear', align_corners=False) # [n, 1, 240, 320]
X = X / 255.0 - 0.5
Y = torch.Tensor([i[3][4] for i in dataset]).unsqueeze(-1).to(device) # [n] -> [n,1]

torch.save((X, Y), 'data.pt')