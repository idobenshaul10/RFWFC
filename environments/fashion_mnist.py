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
from models.vgg import *
from torchvision import datasets, transforms
from models.fashion_mnist_model import FashionMnistModel
# https://www.arunprakash.org/2018/12/cnn-fashion-mnist-dataset-pytorch.html

class fashion_mnist(BaseEnviorment):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        if checkpoint_path is None:
            self.model_path = r"C:\projects\RFWFC\results\DL_layers\trained_models\fahsion_mnist_model\weights.10.h5"
        else:
            self.model_path = checkpoint_path

    def get_dataset(self):
        dataset = torchvision.datasets.FashionMNIST(
            root = r'C:\datasets\fashion_mnist',
            train = True,
            download = True,
            transform = transforms.Compose([
                transforms.ToTensor()                                 
            ])
        )
        return dataset            

    def get_layers(self, model):        
        layers = [module for module in model.modules() if type(module) != nn.Sequential][1:]        
        return layers

    def get_model(self):
        model = FashionMnistModel()
        if self.use_cuda:
            model = model.cuda()
        checkpoint = torch.load(self.model_path)['checkpoint']
        model.load_state_dict(checkpoint)
        return model

