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
from models.fashion_mnist_bad_net import fashion_mnist_bad_net

class fashion_mnist_bad(BaseEnviorment):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        self.model_path = None
        if checkpoint_path is not None:            
            self.model_path = checkpoint_path

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
        feature_layers = np.array([module for module in \
            model.feature_extractor.modules() if type(module) != nn.Sequential])
        
        feature_layers = list(feature_layers[[0, 2, 3, 5]])

        layers = feature_layers + [model.FC_network_1, model.FC_network_2, \
            model.Second_Conv_network, model.FC_network_3, model.FC_network_4]        
        return layers


    def get_eval_transform(self):
        transform = transforms.Compose([transforms.Resize((28, 28)),
            transforms.ToTensor()])
        return transform

    def get_model(self, **kwargs):
        model = fashion_mnist_bad_net(**kwargs)
        if self.use_cuda:
            model = model.cuda()
        if self.model_path is not None:
            checkpoint = torch.load(self.model_path)['checkpoint']
            model.load_state_dict(checkpoint)
        return model

