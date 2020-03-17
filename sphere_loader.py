import numpy as np
import pandas as pd
import os
from sklearn import tree, linear_model, ensemble
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix, lil_matrix
import operator
import code
import argparse
import logging
from collections import Counter
from tqdm import tqdm
import math
from global_values import ConstantValues
from tqdm import tqdm 
import logging

class PointGenerator:
	ratios_sample = ConstantValues.ratios
	def __init__(self, dim=2, cube_length=1.25, seed=1, add_noisy_channels=False):		
		self.add_noisy_channels = add_noisy_channels
		self.num_points = 1000000
		self.dim = dim 		
		self.seed = seed
		np.random.seed(self.seed)
		self.cube_length = PointGenerator.get_cube_length_from_dim(self.dim)


		print(f"cube_length:{self.cube_length} for dim:{self.dim}")		
		__, self.points, self.labels = self.make_dataset()

	# what do we demand of the sampling? it is much more probable to get 0 label..
	def get_label(self, vec):
		vec = vec[0]
		norm = np.sqrt(vec.dot(vec))
		return int(norm <= 1)

	def get_random_point_in_cube(self):		
		random_point_in_unit_cube = 2 * np.random.rand(1, self.dim)
		random_point_in_unit_cube -= 1
		random_point_in_unit_cube *= (self.cube_length/2)
		return random_point_in_unit_cube

	def get_data_point(self):				
		N = 8
		vec = self.get_random_point_in_cube()
		label= self.get_label(vec)
		if self.add_noisy_channels: 
			noisy = np.random.rand(1,N)
			vec = np.expand_dims(np.append(vec, noisy), axis=0)
		return vec, label

	def __getitem__(self, index):
		if index > len(self.points):
			print("requested datasize is not available!")
			exit()

		# indices = np.random.choice(len(self.points), index, replace=False)
		indices = np.arange(index)
		result = self.points[indices], self.labels[indices]	
		return result

	def make_dataset(self):
		points, labels = [], []
		for i in tqdm(range(self.num_points)):
			new_point, new_label = self.get_data_point()
			points.append(new_point)
			labels.append(new_label)

		labels_counter = Counter(labels)
		logging.info(f"LABELS COUNTER: {labels_counter}")
		logging.info(f"cube vol:{pow(self.cube_length, self.dim)}, \
			ball_vol:{self.get_n_ball_volume(self.dim)}")
		logging.info("done making dataset")				
		points = np.array(points).squeeze()		
		labels = np.array(labels).reshape(-1, 1)

		return labels_counter, points, labels

	@staticmethod
	def get_cube_length_from_dim(n):
		n_ball_volume = PointGenerator.get_n_ball_volume(n)		
		# we want C*V~q^n - I found C empiraclly to balance datasets
		C = PointGenerator.ratios_sample[n]
		q = pow((C*n_ball_volume), 1/n)
		return q

	@staticmethod
	def get_n_ball_volume(n):
		n = float(n)
		volume = math.pi**(n/2)/math.gamma(n/2 + 1)
		return volume


if __name__ == '__main__':
	pointGen = PointGenerator(dim=50)
	__, dataset, labels = pointGen.make_dataset()
	print(f'dataset:{Counter(labels)}')	
	# print(f'dataset[0].shape:{dataset[0].shape}')