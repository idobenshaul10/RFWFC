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
from models.vgg import VGG
# https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320

class cifar10(BaseEnviorment):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        self.model_path = None
        if checkpoint_path is not None:            
            self.model_path = checkpoint_path

    def get_dataset(self):
        dataset = datasets.CIFAR10(root=r'C:\datasets\cifar10', 
           train=True, 
           transform=self.get_eval_transform(),
           download=True)
        return dataset

    def get_test_dataset(self):
        dataset = datasets.CIFAR10(root=r'C:\datasets\cifar10', 
           train=False, 
           transform=self.get_eval_transform(),
           download=True)        
        return dataset

    def get_eval_transform(self):
        transform = transforms.Compose([transforms.Resize((32, 32)),        	
            transforms.ToTensor()])
        return transform

    def get_layers(self, model):        
        # layers = [model.layer1, model.layer2, model.layer3, model.layer4, model.avg_pool]
        layers = [k for k in model.features if type(k) == nn.MaxPool2d]
        return layers

    def get_model(self, **kwargs):
        model = VGG('VGG16')        
        if self.use_cuda:
            model = model.cuda()
        if self.model_path is not None:
            checkpoint = torch.load(self.model_path)['checkpoint']
            model.load_state_dict(checkpoint)
        return model



    # def get_layers(self, model):        
    #     layers = [model.max_pool, model.layer2, model.layer3, model.fc1, model.fc2]
    #     return layers

    # def get_model(self, **kwargs):
    #     model = cifar10_net(**kwargs)        
    #     if self.use_cuda:
    #         model = model.cuda()
    #     if self.model_path is not None:
    #         checkpoint = torch.load(self.model_path)['checkpoint']
    #         model.load_state_dict(checkpoint)
    #     return model