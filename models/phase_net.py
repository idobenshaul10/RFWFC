import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.fft as fft

class phase_net(nn.Module):
    def __init__(self, **kwargs):
        super(phase_net, self).__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)        
        
        self.fc1 = nn.Linear(in_features=784, out_features=1024)        
        self.fc2 = nn.Linear(in_features=1024, out_features=2*28**2)
        self.fc3 = nn.Linear(in_features=2*28**2, out_features=2*28**2) 
        
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1, padding=1)
        self.conv_layer2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, padding=1)
        self.conv_layer3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.fc_output = nn.Linear(in_features=784, out_features=10)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ReLU = nn.ReLU()

    def get_image_from_phase(self, mag, dft):        
        real, imag = dft[:,:,:, 0], dft[:,:,:, 1]        
        back = torch.complex(real, imag)        
        img_back = fft.ifftn(back)
        img_back = img_back.abs()
        return img_back
        
    def forward(self, mag):
        batch_size = mag.size(0)
        # out = mag.view(batch_size, -1)
        # out = self.fc1(out)
        # out = self.ReLU(out)        
        # out = self.fc2(out)
        # out = self.ReLU(out)
        # out = self.fc3(out)        
        # dft_pred = out.view(batch_size, 28, 28, 2)
        # img = self.get_image_from_phase(mag, dft_pred).unsqueeze(1)
        
        out = self.conv_layer1(mag)
        out = self.ReLU(out)
        out = self.max_pool(out)        
        out = self.conv_layer2(out)
        out = self.ReLU(out)
        out = self.conv_layer3(out)
        out = self.ReLU(out)
        out = mag.view(batch_size, -1)
        out = self.fc_output(out)
        return out, img


if __name__ == '__main__':  
    a = torch.rand(128, 28, 28)
    model = phase_net()
    out, img = model(a)
    import pdb; pdb.set_trace()