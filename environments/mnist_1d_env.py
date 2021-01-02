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
from datasets.mnist_1d import Mnist1DDataset
from models.mnist_1d_net import mnist_1d_net

class mnist_1d_env(BaseEnviorment):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        self.model_path = None
        if checkpoint_path is not None:            
            self.model_path = checkpoint_path

    def get_dataset(self):
        dataset = Mnist1DDataset(train=True)
        return dataset

    def get_test_dataset(self):
        dataset = Mnist1DDataset(train=False)
        return dataset

    def get_eval_transform(self):
        transform = transforms.Compose([transforms.ToTensor()])
        return transform

    def get_layers(self, model):        
        layers = [model.bn_1, model.bn_2, model.bn_3, model.bn_4, model.bn_5]
        # layers = [model.conv1, model.conv2, model.conv3, model.conv4, model.conv5, model.linear]
        return layers

    def get_model(self, **kwargs):
        model = mnist_1d_net(**kwargs)
        if self.use_cuda:
            model = model.cuda()
        if self.model_path is not None:        
            checkpoint = torch.load(self.model_path)['checkpoint']
            model.load_state_dict(checkpoint)
        return model