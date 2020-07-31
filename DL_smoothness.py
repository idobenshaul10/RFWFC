from __future__ import print_function
import os 
import sys
import argparse
import numpy as np
from tqdm import tqdm
import torch
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
from utils import *
import time

ion()

def get_args():
	parser = argparse.ArgumentParser(description='Network Smoothness Script')
	# regressor args
	parser.add_argument('--trees',default=1,type=int,help='Number of trees in the forest.')	
	parser.add_argument('--depth', default=15, type=int,help='Maximum depth of each tree.Use 0 for unlimited depth.')		
	parser.add_argument('--criterion',default='gini',help='Splitting criterion.')
	parser.add_argument('--bagging',default=0.8,type=float,help='Bagging. Only available when using the "decision_tree_with_bagging" regressor.')
	parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')	
	parser.add_argument('--output_folder', type=str, default=r"C:\projects\RFWFC\results\DL_layers", \
						help='path to save results')
	parser.add_argument('--num_wavelets', default=2000, type=int,help='# wavelets in N-term approx')	
	parser.add_argument('--batch_size', type=int, default=1024)
	parser.add_argument('--env_name', type=str, default="cifar10")
	parser.add_argument('--epsilon_1', type=float, default=0.01)

	args = parser.parse_args()
	args.epsilon_2 = 3*args.epsilon_1
	return args

args = get_args()
BATCH_SIZE = args.batch_size
use_cuda = torch.cuda.is_available()

m = '.'.join(['environments', args.env_name])
module = importlib.import_module(m)
dict_input = vars(args)
environment = eval(f"module.{args.env_name}()")

model, dataset, layers = environment.load_enviorment()
data_loader = torch.utils.data.DataLoader(dataset,
	batch_size=BATCH_SIZE,
	shuffle=True)


activation = {}
def get_activation(name):
	def hook(model, input, output):		
		try:
			new_outputs = output.detach().view(BATCH_SIZE, -1).cpu()
		except:
			import pdb; pdb.set_trace()
			new_outputs = output.detach().view(10, -1).cpu()

		if name not in activation:
			activation[name] = new_outputs
		else:				
			try:	
				activation[name] = \
					torch.cat((activation[name], new_outputs), dim=0)
			except:				
				new_outputs = output.detach().view(-1, activation[name].shape[1]).cpu()
				activation[name] = \
					torch.cat((activation[name], new_outputs), dim=0)
		
	return hook

Y = torch.cat([target for (data, target) in tqdm(data_loader)]).detach()
N_wavelets = 10000
norm_normalization = 'num_samples'
model.eval()
sizes, alphas = [], []

args.output_folder = os.path.join(args.output_folder, \
	f"{args.env_name}_{args.trees}_{args.depth}_{args.epsilon_1}_{args.epsilon_2}")
if not os.path.isdir(args.output_folder):
	os.mkdir(args.output_folder)

with torch.no_grad():
	for k, layer in enumerate(layers):
		# if k <= 1:
		# 	continue
		print(f"LAYER {k}")
		if type(layer) == torch.nn.modules.AvgPool2d or type(layer) == torch.nn.Linear:
		# if type(layer) == torch.nn.modules.pooling.MaxPool2d or type(layer) == torch.nn.Linear:
			if type(layer) == torch.nn.modules.pooling.MaxPool2d:
				layer_str = "MaxPool"
			if type(layer) == torch.nn.Linear:
				layer_str = "Linear"
			if type(layer) == torch.nn.ReLU:
				layer_str = "ReLU"

			layer_name = f'layer_{k}'
			handle = layer.register_forward_hook(get_activation(layer_name))
			for i, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader)):	
				if use_cuda:
					data = data.cuda()
				import pdb; pdb.set_trace()
				model(data)

			X = activation[list(activation.keys())[0]]
			start = time.time()
			Y = np.array(Y).reshape(-1, 1)
			X = np.array(X).squeeze()			
			assert(Y.shape[0] == X.shape[0])
			print(f"Y shape:{Y.shape}")
			alpha_index, __, __, __, __ = run_alpha_smoothness(X, Y, t_method="WF", \
				num_wavelets=N_wavelets, n_trees=args.trees, \
				m_depth=args.depth, \
				n_state=2000, normalize=False, \
				norm_normalization=norm_normalization, error_TH=0., 
				text=f"layer_{k}_{layer_str}", output_folder=args.output_folder, 
				epsilon_1=args.epsilon_1, epsilon_2=args.epsilon_2)

			print(f"alpha for layer {k} is {alpha_index}")			
			handle.remove()
			del activation[layer_name]
			sizes.append(k)
			alphas.append(alpha_index)

plt.figure(1)
plt.clf()
if type(alphas) == list:
	plt.fill_between(sizes, [k[0] for k in alphas], [k[1] for k in alphas], \
		alpha=0.2, facecolor='#089FFF', \
		linewidth=4)
	plt.plot(sizes, [np.array(k).mean()	 for k in alphas], 'k', color='#1B2ACC')
else:
	plt.plot(sizes, alphas, 'k', color='#1B2ACC')

plt.title("VGG ReLu Angle Smoothness")
plt.xlabel(f'dataset size')
plt.ylabel(f'evaluate_smoothnes index- alpha')

save_graph=True
if save_graph:
	save_path = os.path.join(args.output_folder, "result.png")	
	print(f"save_path:{save_path}")
	plt.savefig(save_path, \
		dpi=300, bbox_inches='tight')