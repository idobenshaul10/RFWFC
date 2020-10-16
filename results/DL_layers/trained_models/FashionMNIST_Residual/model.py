import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class ResidualBlock(nn.Module):
    def __init__(self, use_residual, in_channels, out_channels, kernel_size, padding=0):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.use_residual = use_residual

    def forward(self,x):
        residual = x
        out = self.conv(x)
        out = self.batch_norm(out)
        out = F.relu(out)
        if self.use_residual:
            out += residual
        return out

class FashionCNN(nn.Module):
    def __init__(self, use_residual=True):
        super(FashionCNN, self).__init__()
        self.ReLU = nn.ReLU()
        self.use_residual = use_residual
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.layer1 = ResidualBlock(use_residual=self.use_residual, \
            in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.layer2 = ResidualBlock(use_residual=self.use_residual, \
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.layer3 = ResidualBlock(use_residual=self.use_residual, \
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        
        self.drop = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(in_features=32*14*14, out_features=600)        
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
        
    def forward(self, x):
        out = self.layer1(x)
        import pdb; pdb.set_trace()
        out = self.max_pool(out)

        out = self.drop(out)
        out = self.layer2(out)

        out = self.drop(out)
        out = self.layer3(out)        

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.ReLU(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.ReLU(out)
        out = self.fc3(out)
        return out