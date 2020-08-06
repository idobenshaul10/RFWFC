import os
import sys
import argparse
import numpy as np
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, ion, show
from utils import *
import time
from environments.base_environment import *
from models.simple_cifar10_model import Simple_CIFAR10

class cifar10_simple(BaseEnviorment):
	def __init__(self, checkpoint_path):
		super().__init__()
		if checkpoint_path is None:
			self.model_path = r"C:\projects\RFWFC\results\DL_layers\trained_models\cifar10_simple\weights.0.h5"
		else:
			self.model_path = checkpoint_path

	def get_dataset(self):
		dataset = \
			torchvision.datasets.CIFAR10(r"C:\datasets\cifar10", train=True, \
				transform=self.get_eval_transform(), target_transform=None, download=False)
		return dataset

	def get_layers(self, model):		
		layers = [module for module in model.modules()][1:]		
		return layers

	def get_eval_transform(self):
		transform = transforms.Compose(
			[transforms.ToTensor(),
			 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
		return transform

	def get_model(self):
		model = Simple_CIFAR10()
		if self.use_cuda:
			model = model.cuda()		

		checkpoint = torch.load(self.model_path)['checkpoint']
		model.load_state_dict(checkpoint)
		return model

