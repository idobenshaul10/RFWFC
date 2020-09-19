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
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )

        self.FC_network = nn.Sequential(
            nn.Linear(in_features=400, out_features=81),
            nn.ReLU()
        )

        self.Second_Conv_network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=9),
            nn.ReLU()
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
        x = x.view(batch_size, 1, 9, 9)        
        x = self.Second_Conv_network(x)        
        logits = torch.flatten(x, 1)        
        probs = self.softmax(logits)
        return logits, probs





