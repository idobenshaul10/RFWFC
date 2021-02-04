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
import numpy as np
import importlib
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from clustering import kmeans_cluster, get_clustering_statistics
from utils.utils import *
import time
import json
import albumentations as A
from collections import defaultdict
import cv2
from utils.utils import visualize_augmentation
from torch.utils.data import TensorDataset, DataLoader
import glob
import pickle
from DL_smoothness_ensemble import init_params
from sklearn.metrics import *
from pycm import ConfusionMatrix
#  python .\DL_smoothness.py --env_name mnist --checkpoint_path "C:\projects\RFWFC\results\trained_models\weights.80.h5" --use_clustering

ion()

def get_args():
	parser = argparse.ArgumentParser(description='Network Smoothness Script')	
	parser.add_argument('--trees',default=1,type=int,help='Number of trees in the forest.')	
	parser.add_argument('--depth', default=15, type=int,help='Maximum depth of each tree.Use 0 for unlimited depth.')		
	parser.add_argument('--criterion',default='gini',help='Splitting criterion.')
	parser.add_argument('--bagging',default=0.8,type=float,help='Bagging. Only available when using the "decision_tree_with_bagging" regressor.')
	parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')	
	parser.add_argument('--num_wavelets', default=2000, type=int,help='# wavelets in N-term approx')	
	parser.add_argument('--batch_size', type=int, default=256)
	parser.add_argument('--env_name', type=str, default="mnist")
	parser.add_argument('--checkpoints_folder', type=str, default=None)
	parser.add_argument('--high_range_epsilon', type=float, default=0.1)		
	parser.add_argument('--calc_test', action='store_true', default=False)
	parser.add_argument('--output_folder', type=str, default=None)
	parser.add_argument('--checkpoint_file_name', type=str, default="weights.best.h5")



	args = parser.parse_args()
	args.low_range_epsilon = 4*args.high_range_epsilon
	return args


def save_alphas_plot(args, alphas, sizes, test_stats=None, clustering_stats=None):
	plt.figure(1)
	plt.clf()
	if type(alphas) == list:
		plt.fill_between(sizes, [k[0] for k in alphas], [k[-1] for k in alphas], \
			alpha=0.2, facecolor='#089FFF', \
			linewidth=4)
		plt.plot(sizes, [np.array(k).mean()	 for k in alphas], 'k', color='#1B2ACC')
	else:
		plt.plot(sizes, alphas, 'k', color='#1B2ACC')

	plt.title(f"{args.env_name} Angle Smoothness")
	
	acc_txt = ''
	if test_stats is not None and 'top_1_accuracy' in test_stats:
		acc_txt = f"TEST Top1-ACC {test_stats['top_1_accuracy']}"
		
	plt.xlabel(f'Layer\n\n{acc_txt}')
	plt.ylabel(f'evaluate_smoothnes index- alpha')

	save_graph=True
	if save_graph:
		save_path = os.path.join(args.output_folder, "result.png")
		print(f"save_path:{save_path}")
		plt.savefig(save_path, \
			dpi=300, bbox_inches='tight')

	def convert(o):
		if isinstance(o, np.generic): return o.item()
		raise TypeError

	json_file_name = os.path.join(args.output_folder, "result.json")
	write_data = {}
	write_data['alphas'] = [tuple(k) for k in alphas]
	write_data['sizes'] = sizes
	write_data['test_stats'] = test_stats
	write_data['clustering_stats'] = clustering_stats
	norms_path = os.path.join(args.output_folder, json_file_name)	
	with open(norms_path, "w+") as f:
		json.dump(write_data, f, default=convert)

def get_test_outputs(model, data_loader, device):	
	softmax = nn.Softmax(dim=1)	
	y_pred = []
	y_pred_labels = []
	with torch.no_grad():
		model.eval()
		for X, y_true in data_loader:
			X = X.to(device)			
			output = model(X)
			probs = softmax(output).cpu()			
			y_pred.extend(list(probs.cpu().numpy()))
			predicted_labels = torch.max(probs, 1)[1]
			y_pred_labels.extend(list(predicted_labels.cpu().numpy()))
			
	return y_pred, y_pred_labels

def calculate_stats(gt, y_pred, y_pred_labels):
	# stats_names = ['accuracy_score', 'balanced_accuracy_score', 'average_precision_score', 'roc_auc_score']

	gt = np.array(gt)
	y_pred = np.array(y_pred)
	y_pred_labels = np.array(y_pred_labels)
	keys = ["F1", "AUC", "ACC"]
	
	result = {}
	cm = ConfusionMatrix(actual_vector=gt, predict_vector=y_pred_labels)	
	class_stats = cm.class_stat
	for key in keys:	
		values = list(class_stats[key].values())
		# print(f"key:{key}, values:{values}")
		result[key] = np.mean(values)

	result['Overall ACC'] = cm.overall_stat['Overall ACC']
	result['Kappa'] = cm.overall_stat['Kappa']
	
	return result
	# stats_names = ['roc_auc_score']
	# for stat_type in stats_names:
	# 	import pdb; pdb.set_trace()
	# 	value = eval(stat_type)(gt, y_pred)


def get_checkpoint_stats(args, models, test_dataset):	
	for model in models:
		model.eval()
	sizes, alphas = [], []
	clustering_stats = defaultdict(list)

	device = 'cuda' if args.use_cuda else 'cpu'
	with torch.no_grad():	
		test_stats = None
		if args.calc_test and test_dataset is not None:
			test_stats = defaultdict(list)			
			test_loader = torch.utils.data.DataLoader(test_dataset, \
				batch_size=args.batch_size, shuffle=False)
			
			Y_test = torch.cat([target for (data, target) in tqdm(test_loader)]).detach()

			result = defaultdict(list)
			for model in models:
				y_pred, y_pred_labels = get_test_outputs(model, test_loader, device)
				cur_model_stats = calculate_stats(Y_test, y_pred, y_pred_labels)				
				for key, value in cur_model_stats.items():					
					result[key].append(value)

			# import pdb; pdb.set_trace()
			for key in result:
				result[key] = np.mean(result[key])

		print(result)
		# result['top_1_accuracy'] = np.mean(test_accuracy)
		# save_alphas_plot(args, alphas, sizes, test_stats, clustering_stats)

if __name__ == '__main__':
	args, models, dataset, test_dataset, layers, data_loader = init_params()
	get_checkpoint_stats(args, models, test_dataset)