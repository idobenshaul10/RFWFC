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

	data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
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

activation = {}
def get_activation(name, args):
	def hook(model, input, output):		
		if name not in activation:
			activation[name] = output.detach().view(args.batch_size, -1).cpu()
		else:
			try:
				new_outputs = output.detach().view(-1, activation[name].shape[1]).cpu()
				activation[name] = \
					torch.cat((activation[name], new_outputs), dim=0)
			except:
				pass		
	return hook

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

def run_smoothness_analysis(args, models, dataset, test_dataset, layers, data_loader):		
	Y = torch.cat([torch.tensor(np.array(img)) for (_, _, img) in tqdm(data_loader)])
	# Y = torch.cat([torch.tensor(np.array(img)) for (_, _, img) in tqdm(data_loader)]).detach()
	N_wavelets = 10000
	norm_normalization = 'num_samples'
	for model in models:
		model.eval()
	sizes, alphas = [], []
	clustering_stats = defaultdict(list)
	with torch.no_grad():		
		for k in [-1] + list(range(len(layers[0]))):		
			layer_str = 'layer'
			print(f"LAYER {k}, type:{layer_str}")
			layer_name = f'layer_{k}'

			if k == -1:				
				X = torch.cat([data for (data, _, _) in tqdm(data_loader)]).detach()
				X = X.view(X.shape[0], -1)
			else:
				X = []
				for model_idx, model in tqdm(enumerate(models), total=len(models)):
					handle = layers[model_idx][k].register_forward_hook(get_activation(layer_name, args))
					for i, (data, _, _) in tqdm(enumerate(data_loader), total=len(data_loader)):	
						if args.use_cuda:
							data = data.cuda()		
						model(data)
						del data
						# print(activation[list(activation.keys())[0]].shape)

					cur_X = activation[list(activation.keys())[0]]
					X.append(cur_X.cpu().unsqueeze(0))
					handle.remove()
					del activation[layer_name]
				
				X = np.vstack((X))
				X = np.mean(X, axis=0)
			
			start = time.time()			
			# Y = np.array(Y).reshape(-1, 1)
			X = np.array(X).squeeze()
			Y = np.array(Y).reshape(X.shape[0], -1)

			print(f"X.shape:{X.shape}, Y shape:{Y.shape}")
			assert(Y.shape[0] == X.shape[0])		

			if not args.create_umap:
				alpha_index, __, __, __, __ = run_alpha_smoothness(X, Y, t_method="WF", \
					num_wavelets=N_wavelets, n_trees=args.trees, \
					m_depth=args.depth, \
					n_state=args.seed, normalize=False, \
					norm_normalization=norm_normalization, error_TH=0., 
					text=f"layer_{k}_{layer_str}", output_folder=args.output_folder, 
					epsilon_1=args.high_range_epsilon, epsilon_2=args.low_range_epsilon)
				
				print(f"ALPHA for LAYER {k} is {np.mean(alpha_index)}")
				if args.use_clustering:
					kmeans = kmeans_cluster(X, Y, False, args.output_folder, f"layer_{k}")
					clustering_stats[k] = get_clustering_statistics(X, Y, kmeans)

				sizes.append(k)
				alphas.append(alpha_index)
			else:
				kmeans_cluster(X, Y, True, args.output_folder, f"layer_{k}")

	if not args.create_umap:	
		test_stats = None		
		test_stats = {}		
		save_alphas_plot(args, alphas, sizes, test_stats, clustering_stats)

if __name__ == '__main__':
	args, models, dataset, test_dataset, layers, data_loader = init_params()	
	run_smoothness_analysis(args, models, dataset, test_dataset,  layers, data_loader)