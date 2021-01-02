import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class mnist_1d_net(nn.Module):
    def __init__(self, channels=80, linear_in=160, **kwargs):
        super(mnist_1d_net, self).__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.conv1 = nn.Conv1d(1, channels, 5, stride=2, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(channels, channels, 3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(channels, channels, 3, stride=2, padding=1)
        self.conv5 = nn.Conv1d(channels, channels, 3, stride=2, padding=1)
        self.activation = nn.Sigmoid()
        self.bn_1 = nn.BatchNorm1d(channels)
        self.bn_2 = nn.BatchNorm1d(channels)
        self.bn_3 = nn.BatchNorm1d(channels)
        self.bn_4 = nn.BatchNorm1d(channels)
        self.bn_5 = nn.BatchNorm1d(channels)

        self.linear = nn.Linear(linear_in, 10)
        print("Initialized mnist_1d_net model with {} parameters".format(self.count_params()))

    def count_params(self):
        return sum([p.view(-1).shape[0] for p in self.parameters()])

    def forward(self, x, verbose=False): # the print statements are for debugging        
        x = x.view(-1,1,x.shape[-1])
        out = self.conv1(x)
        out = self.activation(out)
        out = self.bn_1(out)

        out = self.conv2(out)
        out = self.activation(out)
        out = self.bn_2(out)

        out = self.conv3(out)
        out = self.activation(out)
        out = self.bn_3(out)

        out = self.conv4(out)
        out = self.activation(out)
        out = self.bn_4(out)

        out = self.conv5(out)
        out = self.activation(out)
        out = self.bn_5(out)
        out = out.view(out.shape[0], -1)           

        return self.linear(out)
