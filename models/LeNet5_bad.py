import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class LeNet5Bad(nn.Module):
    def __init__(self, n_classes):
        super(LeNet5Bad, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2)
        )

        self.FC_network = nn.Sequential(
            nn.Linear(in_features=9, out_features=16),
            nn.Tanh()
        )

        self.Second_Conv_network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=4),
            nn.Tanh()
        )

        if torch.cuda.is_available():
            self.feature_extractor = self.feature_extractor.cuda()
            self.FC_network = self.FC_network.cuda()
            self.Second_Conv_network = self.Second_Conv_network.cuda()

    def forward(self, x):        
        batch_size, _, _, _ = x.shape        
        x = self.feature_extractor(x)       
        x = torch.flatten(x, 1)        
        x = self.FC_network(x)
        x = x.view(batch_size, 1, 4, 4) 
        x = self.Second_Conv_network(x)
        logits = torch.flatten(x, 1)     
        probs = self.softmax(logits)
        return logits





