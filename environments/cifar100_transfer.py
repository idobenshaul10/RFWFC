import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, ion, show
from utils import *
import time
from environments.base_environment import *
from torchvision import datasets, transforms
from models.cifar10_net import cifar10_net
from models.resnet import ResNet18
from models.cifar100_transfer_net import cifar100_transfer_net

class cifar100_transfer(BaseEnviorment):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        self.model_path = None
        if checkpoint_path is not None:            
            self.model_path = checkpoint_path

    def get_dataset(self):
        dataset = datasets.CIFAR100(root=r'C:\datasets\cifar100', 
           train=True, 
           transform=self.get_train_eval_transform(),
           download=True)        
        return dataset

    def get_test_dataset(self):
        dataset = datasets.CIFAR100(root=r'C:\datasets\cifar100', 
           train=False, 
           transform=self.get_test_eval_transform(),
           download=True)        
        return dataset

    def get_train_eval_transform(self):
        transform = transforms.Compose([transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return transform

    def get_test_eval_transform(self):
        transform = transforms.Compose([transforms.Resize((224, 224)),
            transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return transform

    def get_layers(self, model):        
        layers = [block for block in model._modules['blocks']][9:]
        return layers

    def get_model(self, **kwargs):
        import timm
        # model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True, force_reload=True)        
        model = torch.load(r"C:\projects\DL_Smoothness_Results\transformers\torch_model.h5")
        # model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224-a1311bcf', pretrained=True)        
        
        if self.use_cuda:
            model = model.cuda()        
        return model