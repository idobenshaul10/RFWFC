from __future__ import print_function
import os 
import sys
import argparse
import torch
import numpy as np
from numpy import pi
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
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, ion, show
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
import sklearn.datasets as datasets
import pickle
import seaborn as sns
from sklearn.metrics import roc_auc_score

#  python .\DL_smoothness.py --env_name mnist --checkpoint_path "C:\projects\RFWFC\results\trained_models\weights.80.h5" --use_clustering
# plt.style.use('dark_background')
ion()
np.random.seed(2)

def get_args():
	parser = argparse.ArgumentParser(description='Network Smoothness Script')   
	parser.add_argument('--trees',default=1,type=int,help='Number of trees in the forest.') 
	parser.add_argument('--depth', default=10, type=int,help='Maximum depth of each tree.Use 0 for unlimited depth.')       
	parser.add_argument('--criterion',default='gini',help='Splitting criterion.')
	parser.add_argument('--bagging',default=0.8,type=float,help='Bagging. Only available when using the "decision_tree_with_bagging" regressor.')
	parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')    
	parser.add_argument('--output_folder', type=str, default=r"C:\projects\DL_Smoothness_Results\non_linear_synthetic", \
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

def compare_methods(args, X, Y, verbose=False):
	criterion = nn.CrossEntropyLoss()
	N_wavelets = 10000
	norm_normalization = 'num_samples'	
	alpha_index, __, __, __, __ = run_alpha_smoothness(X, Y, t_method="WF", \
		num_wavelets=N_wavelets, n_trees=args.trees, \
		m_depth=args.depth, \
		n_state=args.seed, normalize=False, \
		norm_normalization=norm_normalization, error_TH=0., 
		text=f"Alpha Smoothness", output_folder=args.output_folder, 
		epsilon_1=args.high_range_epsilon, epsilon_2=args.low_range_epsilon)
	
	smoothness = alpha_index.mean()
	kmeans = kmeans_cluster(X, Y, False, args.output_folder, f"synthetic")
	clustering_stats = get_clustering_statistics(X, Y, kmeans)

	lm = linear_model.LogisticRegression()
	lm.fit(X, Y)
	y_preds = lm.predict_proba(X)[:, 1]
	y_cat_pres = lm.predict(X)
	
	if verbose:
		colors = ['magenta', 'dodgerblue']
		plt.scatter(X[:, 0], X[:, 1], marker='o', c=kmeans.predict(X),
					s=25, edgecolor='k', cmap=matplotlib.colors.ListedColormap(colors))
		plt.title('KMeans')
		plt.show(block=True)
		plt.clf()

		plt.scatter(X[:, 0], X[:, 1], marker='o', c=y_cat_pres,
					s=25, edgecolor='k', cmap=matplotlib.colors.ListedColormap(colors))
		plt.title('LP')
		plt.show(block=True)
	
	lp_auc = roc_auc_score(Y, y_preds)
	print(f"AUC:{lp_auc}")
	return clustering_stats, smoothness, lp_auc

def make_spiral_dataset(N = 5000):	
	theta = np.sqrt(np.random.rand(N))*4*pi # np.linspace(0,2*pi,100)

	r_a = 2*theta + pi
	data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
	x_a = data_a + np.random.randn(N,2)	
	y_a = np.ones(N)

	r_b = -2*theta - pi
	data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
	x_b = data_b + np.random.randn(N,2)
	y_b = np.zeros(N)

	X = X = np.row_stack((x_a, x_b))
	Y = np.concatenate((y_a, y_b))	
	return X, Y
	

def get_synthetic_data():
	X, Y = {}, {}
	X['spiral'], Y['spiral'] = make_spiral_dataset()
	# X['moons'], Y['moons'] = datasets.make_moons(n_samples=10000, shuffle=True, noise=0.1, random_state=42)
	X['circles'], Y['circles'] = datasets.make_circles(n_samples=10000, shuffle=True, noise=0.01, random_state=42)
	X['gaussian'], Y['gaussian'] = datasets.make_gaussian_quantiles(n_samples=10000, \
		n_features=2, n_classes=2, random_state=42)
	
	return X, Y


if __name__ == '__main__':
	args = get_args()
	args.use_cuda = torch.cuda.is_available()
	clustering_stats_per_metric = defaultdict(list)
	X, Y = get_synthetic_data()
	verbose = True

	colors = ['magenta', 'dodgerblue']
	for key in X:
		cur_X, cur_Y = X[key], Y[key]
		if verbose:			
			plt.scatter(cur_X[:, 0], cur_X[:, 1], marker='o', c=cur_Y, \
				cmap=matplotlib.colors.ListedColormap(colors), s=25, edgecolor='k')
			plt.show(block=True)

		cur_clustering_stats, cur_smoothness, lp_auc = compare_methods(args, cur_X, cur_Y, verbose)
		for metric, value in cur_clustering_stats.items():
			clustering_stats_per_metric[metric].append(value)
		clustering_stats_per_metric['Linear Probes Train AUC'].append(lp_auc)
		clustering_stats_per_metric['Besov Smoothness'].append(cur_smoothness.mean())


	metrics = list(clustering_stats_per_metric.keys())
	values = np.array([clustering_stats_per_metric[k] for k in metrics])	
	values = values.reshape(-1, len(list(X.keys())))
	dataset_names = ["Spiral", "Circles", "Gaussian Quantiles"]
	

	sns.heatmap(values,  annot=True, vmin=0., vmax=1.,\
		xticklabels=dataset_names, yticklabels=metrics)
	plt.title("Metrics for datasets: Besov-Smoothness vs. Clustering")	
	plt.show(block=True)
	# print(correlations)

	# pickle.dump(clustering_stats, open(r"C:\projects\DL_Smoothness_Results\clustering\clustering_stats.p", "wb"))
	# pickle.dump(smoothness, open(r"C:\projects\DL_Smoothness_Results\clustering\smoothness.p", "wb"))
