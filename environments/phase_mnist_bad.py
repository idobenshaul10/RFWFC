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
from datasets.mnist_phase_retrival import MnistPhaseDataset
from torchvision import datasets, transforms
from models.cifar10_net import cifar10_net
from models.resnet import ResNet18
from models.vgg import VGG
from models.phase_net_bad import phase_net_bad

class phase_mnist_bad(BaseEnviorment):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        self.model_path = None
        if checkpoint_path is not None:            
            self.model_path = checkpoint_path

    def get_dataset(self):
        dataset = MnistPhaseDataset(train=True)
        return dataset

    def get_test_dataset(self):
        dataset = MnistPhaseDataset(train=False)
        return dataset

    def get_layers(self, model):        
        layers = [model.layer1, model.layer2, model.layer3, model.layer4]
        # layers = [model.fc1, model.fc2, model.fc3]
        return layers

    def get_model(self, **kwargs):
        model = phase_bad_net(**kwargs)        
        if self.use_cuda:
            model = model.cuda()        
        return model