import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class phase_net(nn.Module):
    def __init__(self, **kwargs):
        super(phase_net, self).__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)        
        
        self.fc1 = nn.Linear(in_features=784, out_features=1024)         
        # self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=2*28**2)
        self.fc3 = nn.Linear(in_features=2*28**2, out_features=2*28**2) 
        self.ReLU = nn.ReLU()
        
    def forward(self, x):
        batch_size = x.size(0)
        out = x.view(batch_size, -1)        
        out = self.fc1(out)
        out = self.ReLU(out)        
        out = self.fc2(out)
        out = self.ReLU(out)
        out = self.fc3(out)
        # out = self.ReLU(out)
        # out = self.fc4(out)
        out = out.view(batch_size, 28, 28, 2)
        return out