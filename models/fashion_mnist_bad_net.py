import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class fashion_mnist_bad_net(nn.Module):
    def __init__(self, **kwargs):
        super(fashion_mnist_bad_net, self).__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.softmax = nn.Softmax(dim=1)
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2)
        )

        self.FC_network_1 = nn.Sequential(
            nn.Linear(in_features=4, out_features=16),
            nn.Tanh()            
        )

        self.FC_network_2 = nn.Sequential(
            nn.Linear(in_features=16, out_features=16),
            nn.Tanh()
        )

        self.Second_Conv_network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=4),
            nn.Tanh()
        )

        self.FC_network_3 = nn.Sequential(
            nn.Linear(in_features=10, out_features=10),
            nn.Tanh()
        )

        self.FC_network_4 = nn.Sequential(
            nn.Linear(in_features=10, out_features=10),
            nn.Tanh()
        )

        if torch.cuda.is_available():
            self.feature_extractor = self.feature_extractor.cuda()
            self.FC_network_1 = self.FC_network_1.cuda()
            self.FC_network_2 = self.FC_network_2.cuda()
            self.FC_network_3 = self.FC_network_3.cuda()
            self.FC_network_4 = self.FC_network_4.cuda()
            self.Second_Conv_network = self.Second_Conv_network.cuda()
            

    def forward(self, x):        
        batch_size, _, _, _ = x.shape
        
        x = self.feature_extractor(x)      
        x = torch.flatten(x, 1)
        x = self.FC_network_1(x)
        x = self.FC_network_2(x)
        x = x.view(batch_size, 1, 4, 4)
        x = self.Second_Conv_network(x)
        x = torch.flatten(x, 1)        
        x = self.FC_network_3(x)
        x = self.FC_network_4(x)
        logits = torch.flatten(x, 1)
        probs = self.softmax(logits)
        return logits





