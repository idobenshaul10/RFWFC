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
from models.LeNet5_bad import LeNet5Bad

class fashion_mnist_bad(BaseEnviorment):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        if checkpoint_path is None:
            self.model_path = r"C:\projects\RFWFC\results\DL_layers\trained_models\fahsion_mnist_model\weights.0.h5"
        else:
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
        feature_layers = np.array([module for module in model.feature_extractor.modules() if type(module) != nn.Sequential])
        FC_network_layers = np.array([module for module in model.FC_network.modules() if type(module) != nn.Sequential])
        Second_Conv_network_layers = np.array([module for module in model.Second_Conv_network.modules() if type(module) != nn.Sequential])
        
        feature_layers = list(feature_layers[[2, 5]])
        FC_network_layers = list(FC_network_layers[[1]])        
        Second_Conv_network_layers = list(Second_Conv_network_layers[[1]]) 
        layers = feature_layers + FC_network_layers + Second_Conv_network_layers + [model.softmax]        
        return layers

    def get_eval_transform(self):
        transform = transforms.Compose([transforms.Resize((32, 32)),            
            transforms.ToTensor()])
        return transform

    def get_model(self):
        model = LeNet5Bad(10)
        if self.use_cuda:
            model = model.cuda()
        checkpoint = torch.load(self.model_path)['checkpoint']
        model.load_state_dict(checkpoint)
        return model
