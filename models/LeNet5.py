import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class LeNet5(nn.Module):
    def __init__(self, n_classes):
        super(LeNet5, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=256, out_features=84),
            nn.ReLU(),

            nn.Linear(in_features=84, out_features=n_classes),
        )
        if torch.cuda.is_available():
            self.feature_extractor = self.feature_extractor.cuda()
            self.classifier = self.classifier.cuda()

    def forward(self, x):        
        x = self.feature_extractor(x)        
        x = torch.flatten(x, 1)        
        logits = self.classifier(x)
        # probs = self.softmax(logits)
        return logits