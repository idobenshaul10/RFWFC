import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, ion, show
from utils import *
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision 
from torchvision import datasets, transforms

class BaseEnviorment():
	def __init__(self):		
		self.use_cuda = torch.cuda.is_available()

	def load_enviorment(self):
		dataset = self.get_dataset()		
		model = self.get_model()		
		layers = self.get_layers(model)
		return model, dataset, layers

	def get_layers(self):
		pass
	
	def get_dataset(self):
		pass
	
	def get_eval_transform(self):
		pass
	
	def get_model(self):
		pass
