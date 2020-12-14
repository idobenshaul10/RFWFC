import numpy as np
import pickle
import torch
import os

class Mnist1DDataset(torch.utils.data.Dataset):
	def __init__(self, train=True):
		self.train = train
		self.home_dir = r"/home/ido/datasets/mnist_1d"
		self.dataset_path = os.path.join(self.home_dir, "mnist_1d.pkl")
		
		with open(self.dataset_path, 'rb') as handle:
		    self.data = pickle.load(handle)		    

		if self.train:
			self.x = self.data['x']
			self.y = self.data['y']

		else:			
			self.x = self.data['x_test']
			self.y = self.data['y_test']

	def __len__(self):
		return len(self.x)	

	
	def __getitem__(self, idx):
		return torch.tensor(self.x[idx], dtype=torch.float), torch.tensor(self.y[idx])


		

