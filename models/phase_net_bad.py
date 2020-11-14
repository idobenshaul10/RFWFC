import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class phase_net_bad(nn.Module):
    def __init__(self, **kwargs):
        super(phase_net_bad, self).__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)        
        
        self.layer1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.layer2 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.layer3 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)        
        self.ReLU = nn.ReLU()
        
    def forward(self, x):
        batch_size = x.size(0)
        # out = x.view(batch_size, -1)
        out = self.layer1(out)
        out = self.ReLU(out)
        out = self.layer2(out)
        out = self.ReLU(out)
        out = self.layer3(out)

        # out = self.ReLU(out)
        # out = self.fc4(out)
        import pdb; pdb.set_trace()
        out = out.view(batch_size, 28, 28, 2)
        return out