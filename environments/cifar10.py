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

class cifar10(BaseEnviorment):
	def __init__(self):
		super().__init__()
		self.model_path = r"C:\projects\LinearEstimators\best_vgg.pth"		

	def get_dataset(self):
		dataset = \
			torchvision.datasets.CIFAR10(r"C:\datasets\cifar10", train=True, \
				transform=self.get_eval_transform(), target_transform=None, download=False)
		return dataset

	def get_eval_transform(self):
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])
		transform = transforms.Compose([			
				transforms.ToTensor(),
				normalize,
			])
		return transform

	def get_model(self):
		model = vgg19()
		model.features = torch.nn.DataParallel(model.features)
		if self.use_cuda:
			model = model.cuda()

		checkpoint = torch.load(self.model_path)['state_dict']
		model.load_state_dict(checkpoint)
		return model

