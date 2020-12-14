import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

class cifar100_transfer_net(nn.Module):
    def __init__(self, **kwargs):
        super(cifar100_transfer_net, self).__init__()        
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.model = models.vgg16(pretrained=True)
        # self.model = models.densenet121(pretrained=True)
        # self.model = models.resnet18(pretrained=True)
        # self.model = models.resnet50(pretrained=True)        
        # self.model = models.resnet101(pretrained=True)
        # self.model = models.googlenet(pretrained=True)
        # self.batch_norm_1 = nn.BatchNorm1d(512)
        # self.batch_norm_2 = nn.BatchNorm1d(256)


        for param in self.model.parameters():
            param.requires_grad = False
        
        #resnet18/Googlenet
        # n_inputs = self.model.fc.in_features
        # self.model.fc = nn.Sequential(
        #     nn.Linear(n_inputs, 512), nn.ReLU(), nn.Dropout(0.3), self.batch_norm_1,
        #     nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), self.batch_norm_2,
        #     nn.Linear(256, 100))
        # self.model.fc = nn.Sequential(
        #     nn.Linear(n_inputs, 100))

        #densenet121        
        # n_inputs = self.model.classifier.in_features
        # self.model.classifier = nn.Sequential(
        #     nn.Linear(n_inputs, 512), nn.ReLU(), nn.Dropout(0.4),
        #     nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
        #     nn.Linear(256, 100))
        # self.model.classifier = nn.Sequential(
        #     nn.Linear(n_inputs, 100))

        #VGG16
        n_inputs = self.model.classifier[6].in_features
        # self.model.classifier[6] = nn.Sequential(
        #     nn.Linear(n_inputs, 512), nn.ReLU(), nn.Dropout(0.4),
        #     nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
        #     nn.Linear(256, 100))
        self.model.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 100))
        
    def forward(self, x):
        out = self.model(x)
        return out
        