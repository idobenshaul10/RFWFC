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
from models.fashion_mnist_model import fashion_mnist_model
from models.resnet_altered import ResNet18_altered

class fashion_mnist(BaseEnviorment):
    def __init__(self, checkpoint_path=None, use_residual=False):
        super().__init__()
        self.model_path = None
        if checkpoint_path is None:        
            self.model_path = checkpoint_path
        self.use_residual = use_residual

    def get_dataset(self):
        dataset = torchvision.datasets.FashionMNIST(
            root = r'C:\datasets\fashion_mnist',
            train = True,
            download = True,
            transform=self.get_eval_transform()
        )
        return dataset            

    def get_test_dataset(self):
        dataset = torchvision.datasets.FashionMNIST(
            root=r'C:\datasets\fashion_mnist', 
            train=False, 
            transform=self.get_eval_transform(),
            download=True)

        return dataset

    def get_layers(self, model):
        # layers = [model.max_pool, model.layer2, model.layer3, model.fc1, model.fc2]
        layers = [model.layer1, model.layer2, model.layer3, model.layer4, model.avg_pool]
        return layers

    def get_eval_transform(self):
        transform = transforms.Compose([transforms.Resize((28, 28)),            
            transforms.ToTensor()])
        return transform

    def get_model(self, **kwargs):
        # model = fashion_mnist_model(**kwargs)
        model = ResNet18_altered()
        if self.use_cuda:
            model = model.cuda()
        if self.model_path is not None:
            checkpoint = torch.load(self.model_path)['checkpoint']
            model.load_state_dict(checkpoint)
        return model

