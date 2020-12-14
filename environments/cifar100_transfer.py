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
        dataset = datasets.CIFAR100(root=r'/home/ido/datasets/cifar100', 
           train=True, 
           transform=self.get_train_eval_transform(),
           download=True)
        return dataset

    def get_test_dataset(self):
        dataset = datasets.CIFAR100(root=r'/home/ido/datasets/cifar100', 
           train=False, 
           transform=self.get_test_eval_transform(),
           download=True)        
        return dataset

    def get_train_eval_transform(self):
        transform = transforms.Compose([transforms.Resize((32, 32)),            
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return transform

    def get_test_eval_transform(self):
        transform = transforms.Compose([transforms.Resize((32, 32)),            
            transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return transform

    def get_layers(self, model):        
        #VGG16        
        # layers = [[k for k in model.model.features if type(k) == nn.MaxPool2d][-1]] + \
        #     [model.model.classifier[6][2], model.model.classifier[6][5], model.model.classifier[6][6]]
        layers = [[k for k in model.model.features if type(k) == nn.MaxPool2d][-1]] + \
            [model.model.classifier[6]]

        # densenet121
        # layers = [[k for k in model.model.features if type(k) == torchvision.models.densenet._DenseBlock][-1]] + \
        #     [model.model.classifier[2], model.model.classifier[5], model.model.classifier[6]]
        # layers = [[k for k in model.model.features if type(k) == torchvision.models.densenet._DenseBlock][-1]] + \
        #     [model.model.classifier]
        
        #resnet18      
        # layers = [model.model.layer4, model.model.fc[2], model.model.fc[5],  model.model.fc[6]]
        # layers = [model.model.layer4, model.model.fc]
        
        #Googlenet
        # layers = [model.model.inception5b, model.model.fc[2], model.model.fc[5],  model.model.fc[6]]
        # layers = [model.model.inception5b, model.model.fc]

        #inception_v3
        # layers = [model.model.inception5a, model.model.inception5b, model.model.fc]
        return layers

    def get_model(self, **kwargs):
        model = cifar100_transfer_net()
        if self.use_cuda:
            model = model.cuda()
        if self.model_path is not None:
            checkpoint = torch.load(self.model_path)['checkpoint']
            model.load_state_dict(checkpoint)
        return model