from __future__ import print_function
import os 
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision 
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from sklearn import tree, linear_model, ensemble
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, ion, show
import matplotlib as mpl
import numpy as np
import importlib
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from DL_Layer_Analysis.clustering import kmeans_cluster, get_clustering_statistics
from utils.utils import *
import time
import json
import albumentations as A
from collections import defaultdict
import cv2
from utils.utils import visualize_augmentation
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification
import pickle
import seaborn as sns
# plt.style.use('dark_background')
#  python .\DL_smoothness.py --env_name mnist --checkpoint_path "C:\projects\RFWFC\results\trained_models\weights.80.h5" --use_clustering

ion()

def get_args():
	parser = argparse.ArgumentParser(description='Network Smoothness Script')   
	parser.add_argument('--trees',default=1,type=int,help='Number of trees in the forest.') 
	parser.add_argument('--depth', default=15, type=int,help='Maximum depth of each tree.Use 0 for unlimited depth.')       
	parser.add_argument('--criterion',default='gini',help='Splitting criterion.')
	parser.add_argument('--bagging',default=0.8,type=float,help='Bagging. Only available when using the "decision_tree_with_bagging" regressor.')
	parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')    
	parser.add_argument('--output_folder', type=str, default=r"C:\projects\DL_Smoothness_Results\clustering", \
		help='path to save results')
	parser.add_argument('--num_wavelets', default=2000, type=int,help='# wavelets in N-term approx')    
	parser.add_argument('--batch_size', type=int, default=1024)
	parser.add_argument('--env_name', type=str, default="mnist")
	parser.add_argument('--checkpoint_path', type=str, default=None)
	parser.add_argument('--high_range_epsilon', type=float, default=0.1)
	parser.add_argument('--create_umap', action='store_true', default=False)
	parser.add_argument('--use_clustering', action='store_true', default=False)
	parser.add_argument('--calc_test', action='store_true', default=False)

	args = parser.parse_args()
	args.low_range_epsilon = 4*args.high_range_epsilon
	return args

def compare_methods(args, X, Y):
	N_wavelets = 10000	
	norm_normalization = 'num_samples'
	alpha_index, __, __, __, __ = run_alpha_smoothness(X, Y, t_method="WF", \
		num_wavelets=N_wavelets, n_trees=args.trees, \
		m_depth=args.depth, \
		n_state=args.seed, normalize=False, \
		norm_normalization=norm_normalization, error_TH=0., 
		text=f"Alpha Smoothness", output_folder=args.output_folder, 
		epsilon_1=args.high_range_epsilon, epsilon_2=args.low_range_epsilon)

	# print(alpha_index.mean())
	smoothness = alpha_index.mean()
	kmeans = kmeans_cluster(X, Y, False, args.output_folder, f"synthetic")
	clustering_stats = get_clustering_statistics(X, Y, kmeans)

	print(clustering_stats)
	return clustering_stats, smoothness


def get_synthetic_data(num_centers):
	# X, Y = make_classification(n_features=2, n_redundant=0, n_clusters_per_class=3, n_informative=3)
	# plt.cla()
	# fig1, ax1 = plt.subplots()
	centers = np.random.rand(num_centers, 2) * 1000	
	X, Y = make_blobs(n_samples=30000, centers=centers, shuffle=False,
					  random_state=42)

	Y[Y%2 ==0] = 2
	Y[Y%2 ==1] = 1

	# plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y,
	# 			s=25, edgecolor='k')
	# plt.show(block=True)

	return X, Y


if __name__ == '__main__':
	# args = get_args()
	# args.use_cuda = torch.cuda.is_available()
	# clustering_stats, smoothness = {}, {}
	# for num_centers in range(3, 20):
	# 	X, Y = get_synthetic_data(num_centers)
	# 	clustering_stats[num_centers], Smoothness[num_centers] = compare_methods(args, X, Y)
	
	stats = pickle.load( open(r"C:\projects\DL_Smoothness_Results\clustering\clustering_stats.p", "rb" ) )
	smoothness = pickle.load( open( r"C:\projects\DL_Smoothness_Results\clustering\smoothness.p", "rb" ) )
	smoothness_values = [smoothness[k] for k in smoothness.keys()]


	# N=10
	# plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Set1(np.linspace(0,1,N)))
	colors = ['darkblue', 'lightblue', 'dodgerblue', 'navy', 'blue']
	colors_2 = ['gold', 'orange']
	
	for idx, key in enumerate(stats[3].keys()):
		
		values = [stats[i][key]  for i in stats.keys()]		
		# if key == "silhouette_score" or key == "FMI":
		# 	plt.plot(list(stats.keys()), values, label=key, color=colors_2[idx%len(colors_2)])
		# else:
		# 	plt.plot(list(stats.keys()), values, label=key, color=colors[idx])

		sns.lineplot(list(stats.keys()), values, label=key)
		# plt.plot(list(stats.keys()), values, label=key, color=colors_2[idx%len(colors_2)])


	plt.plot(list(smoothness.keys()), smoothness_values, label="Smoothness", color="orchid")
	plt.legend()
	plt.show(block=True)
	
