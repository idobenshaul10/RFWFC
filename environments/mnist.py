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
from models.LeNet5 import LeNet5

class mnist(BaseEnviorment):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        if checkpoint_path is None:
            self.model_path = r"C:\projects\RFWFC\results\DL_layers\trained_models\LeNet5\weights.10.h5"

    def get_dataset(self):
        dataset = datasets.MNIST(root=r'C:\datasets\mnist_data', 
           train=True, 
           transform=self.get_eval_transform(),
           download=True)
        return dataset

    def get_eval_transform(self):
        transform = transforms.Compose([transforms.Resize((32, 32)),
            transforms.ToTensor()])
        return transform

    def get_layers(self, model):
        feature_layers = [module for module in model.feature_extractor.modules() if type(module) != nn.Sequential]
        classifier_layers = [module for module in model.classifier.modules() if type(module) != nn.Sequential]
        layers = feature_layers + classifier_layers
        return layers

    def get_model(self):
        model = LeNet5(10)
        if self.use_cuda:
            model = model.cuda()
        checkpoint = torch.load(self.model_path)['checkpoint']
        model.load_state_dict(checkpoint)
        return model

