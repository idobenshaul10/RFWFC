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

class tiny_image_net(BaseEnviorment):
	def __init__(self, model_name="resnet18"):
		super().__init__()
		self.model_name = model_name

	def get_dataset(self):
		root = r"C:\datasets\imagenet"
		dataset = \
			torchvision.datasets.ImageNet(root, \
				split='train', download=True)
		return dataset

	def get_layers(self, model):
		import pdb; pdb.set_trace()		
		layers = [module for module in model.modules()][1:]		
		return layers

	def get_eval_transform(self):
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

		transform = transforms.Compose(
			[transforms.ToTensor(),
			 normalize])
		return transform

	def get_model(self):
		model = eval(f"{model_name}(pretrained=True)")
		if self.use_cuda:
			model = model.cuda()		
		return model

