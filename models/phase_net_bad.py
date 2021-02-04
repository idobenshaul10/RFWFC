import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.fft as fft

class phase_net_bad(nn.Module):
    def __init__(self, **kwargs):
        super(phase_net_bad, self).__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)        
        
        self.fc1 = nn.Linear(in_features=784, out_features=1024)        
        self.fc2 = nn.Linear(in_features=1024, out_features=2*28**2)
        self.fc3 = nn.Linear(in_features=2*28**2, out_features=2*28**2)

        self.ReLU = nn.ReLU()
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.conv_layer2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1)
        self.conv_layer3 = nn.Conv2d(in_channels=8, out_channels=2, kernel_size=3, padding=1)
        self.fc_output = nn.Linear(in_features=784, out_features=10)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def get_image_from_phase(self, dft):        
        real, imag = dft[:,:,:, 0], dft[:,:,:, 1]        
        back = torch.complex(real, imag)        
        img_back = fft.ifftn(back)        
        img_back = img_back.abs()
        return img_back        

    def forward(self, mag):        
        batch_size = mag.size(0)
        mag = mag.unsqueeze(1)

        out = self.conv_layer1(mag)
        out = self.ReLU(out)
        out = self.max_pool(out)
        out = self.conv_layer2(out)
        out = self.ReLU(out)
        out = self.conv_layer3(out)
        out = self.ReLU(out)
        out = mag.view(batch_size, -1)        
        out = self.fc_output(out)
        return out

if __name__ == '__main__':  
    a = torch.rand(128, 28, 28)
    model = phase_net_bad()
    out = model(a)
    print(out.shape)    