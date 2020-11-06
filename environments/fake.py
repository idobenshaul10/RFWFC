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
from models.fake_model import fake_model

class fake(BaseEnviorment):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        self.model_path = None
        if checkpoint_path is not None:            
            self.model_path = checkpoint_path

    def get_dataset(self):
        dataset = datasets.FakeData(size=60000, image_size=(3, 32, 32),
           transform=self.get_eval_transform())
        return dataset

    def get_test_dataset(self):
        dataset = datasets.FakeData(size=10000, image_size=(3, 32, 32),
           transform=self.get_eval_transform(), random_offset=20)
        return dataset

    def get_eval_transform(self):
        transform = transforms.Compose([transforms.ToTensor()])
        return transform

    def get_layers(self, model):        
        layers = [model.max_pool, model.layer2, model.layer3, model.fc1, model.fc2]
        return layers

    def get_model(self, **kwargs):        
        model = fake_model(**kwargs)

        if self.use_cuda:
            model = model.cuda()
        if self.model_path is not None:
            checkpoint = torch.load(self.model_path)['checkpoint']
            model.load_state_dict(checkpoint)
        return model