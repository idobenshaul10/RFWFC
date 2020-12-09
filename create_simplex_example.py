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
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

#  python test.py --criterion gini --bagging 0.8 --seed 1 --num_wavelets 2000 --batch_size 512 --env_name mnist --high_range_epsilon 0.1 --calc_test --checkpoints_folder C:\projects\DL_Smoothness_Results\trained_models\TWO_LAYER_RESIDUAL\mnist\mnist_2020_10_15-07_59_36_PM --output_folder C:\projects\DL_Smoothness_Results\testing --high_range_epsilon 0.001

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
	parser.add_argument('--create_umap', action='store_true', default=False)
	parser.add_argument('--use_clustering', action='store_true', default=False)
	parser.add_argument('--calc_test', action='store_true', default=False)
	parser.add_argument('--output_folder', type=str, default=None)	

	args = parser.parse_args()
	args.low_range_epsilon = 4*args.high_range_epsilon
	return args

def init_params():	
	args = get_args()
	args.batch_size = args.batch_size
	args.use_cuda = torch.cuda.is_available()

	m = '.'.join(['environments', args.env_name])
	module = importlib.import_module(m)
	dict_input = vars(args)	
	
	environment = eval(f"module.{args.env_name}()")
	folds = glob.glob(os.path.join(args.checkpoints_folder, "*"))	
	folds = [f for f in folds if os.path.isdir(f) and "DL_Analysis" not in f]
	NUM_FOLDS = len(folds)

	params_path = os.path.join(args.checkpoints_folder, 'args.p')
	if os.path.isfile(params_path):
		params = vars(pickle.load(open(params_path, 'rb')))

	__, dataset, test_dataset, __ = environment.load_enviorment()
	models = []
	layers = {}
	for idx, fold in enumerate(sorted(folds)):
		checkpoint_path = os.path.join(fold, "weights.best.h5")		
		cur_model = environment.get_model(**params)
		checkpoint = torch.load(checkpoint_path)['checkpoint']
		cur_model.load_state_dict(checkpoint)
		if torch.cuda.is_available():
			cur_model = cur_model.cuda()
		models.append(cur_model)
		layers[idx] = environment.get_layers(cur_model)

	data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	addition = f"{args.env_name}_{args.trees}_{args.depth}_{args.high_range_epsilon}_{args.low_range_epsilon:.2f}"	
	if args.output_folder is None:
		args.output_folder = os.path.join(args.checkpoints_folder, "DL_Analysis")
	else:
		args.output_folder = os.path.join(args.output_folder, "DL_Analysis")
	if not os.path.isdir(args.output_folder):	
		os.mkdir(args.output_folder)

	args.output_folder = os.path.join(args.output_folder, addition)
	if not os.path.isdir(args.output_folder):	
		os.mkdir(args.output_folder)

	return args, models, dataset, test_dataset, layers, data_loader

def rotate(a=1, b=1, c=1, d=-1):
	theta_cos = c/np.sqrt(a**2+b**2+c**2)
	theta_sin = np.sqrt((a**2+b**2)/(a**2+b**2+c**2))
	u_1 = b/np.sqrt(a**2+b**2+c**2)
	u_2 = -a/np.sqrt(a**2+b**2+c**2)

	result = np.zeros((3,3))
	result[0,0] = theta_cos+(u_1**2)*(1-theta_cos)
	result[0,1] = u_1*u_2*(1-theta_cos)
	result[0,2] = u_2*theta_sin
	result[1,0] = u_1*u_2*(1-theta_cos)
	result[1,1] = theta_cos+(u_2**2)*(1-theta_cos)
	result[1,2] = -u_1*theta_sin
	result[2,0] = -u_2*theta_sin
	result[2,1] = u_1*theta_sin
	result[2,2] = theta_cos

	return result



def draw_three_classes(models, data_loader):
	from scipy.special import softmax
	model = models[0]
	model.eval()
	Y = torch.cat([target for (data, target) in tqdm(data_loader)]).detach()
	while True:
		indices = None
		labels = []
		# classes = [1, 5, 9]
		classes = [0, 6, 2]
		# classes = list(np.random.choice(10, 3, replace=False))
		for i in classes:
			cur_indices = torch.where(Y == i)[0]		
			if indices is None:
				indices = cur_indices
			else:			
				indices = torch.cat((indices, cur_indices))			
			labels.extend([i for _ in range(cur_indices.shape[0])])
			
		preds = []	
		for i, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader)):	
			if args.use_cuda:
				data = data.cuda()		
			y = model(data)
			preds.extend(list(y.detach().cpu().numpy().squeeze()))	
		preds = np.array(preds)	
		preds = preds[indices]
		# while True:		
		# classes = [0, 1, 2]
		
		cur_preds = preds[:, classes]
		cur_preds = softmax(cur_preds, axis=1)
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')
		rand_indices = np.random.choice(len(cur_preds), len(cur_preds), replace=False)	
		shuff_preds = cur_preds[rand_indices]
		labels = np.array(labels)[rand_indices]
		xs, ys, zs = shuff_preds[:, 0], shuff_preds[:, 1], shuff_preds[:, 2]
		

		colors = ["#1b374d", "#ee4f2f", "#fba720"]
		patch_0 = mpatches.Patch(color=colors[0], label=f'{classes[0]}')
		patch_1 = mpatches.Patch(color=colors[1], label=f'{classes[1]}')
		patch_2 = mpatches.Patch(color=colors[2], label=f'{classes[2]}')
		plt.legend(handles=[patch_0, patch_1, patch_2])

		c = [colors[classes.index(k)] for k in labels]	
		ax.scatter(xs, ys, zs, c=c, s=5.)
		plt.show(block=True)
		fig.clf()
	

if __name__ == '__main__':
	args, models, dataset, test_dataset, layers, data_loader = init_params()	
	draw_three_classes(models, data_loader)