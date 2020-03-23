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
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from models.vgg import *
from utils import *
import time

ion()

def get_args():
	parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
	# regressor args
	parser.add_argument('--trees',default=2,type=int,help='Number of trees in the forest.')	
	parser.add_argument('--depth', default=30, type=int,help='Maximum depth of each tree.Use 0 for unlimited depth.')		
	parser.add_argument('--criterion',default='gini',help='Splitting criterion.')
	parser.add_argument('--bagging',default=0.8,type=float,help='Bagging. Only available when using the "decision_tree_with_bagging" regressor.')
	parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
	parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
	parser.add_argument('--output_folder', type=str, default=r"C:\projects\LinearEstimators\results", \
						help='path to save results')
	parser.add_argument('--num_wavelets', default=2000, type=int,help='# wavelets in N-term approx')


	parser.add_argument('--dataset', type=str, default="cifar10")

	args = parser.parse_args()
	return args

BATCH_SIZE = 1024
use_cuda = torch.cuda.is_available()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])

cifar10_dataset = \
	torchvision.datasets.CIFAR10(r"C:\datasets\cifar10", train=True, transform=transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomCrop(32, 4),
			transforms.ToTensor(),
			normalize,
		]), target_transform=None, download=False)
data_loader = torch.utils.data.DataLoader(cifar10_dataset,
	batch_size=BATCH_SIZE,
	shuffle=True)

device = torch.device("cuda" if use_cuda else "cpu")
model = vgg19()
model.features = torch.nn.DataParallel(model.features)
if use_cuda:
	model = model.cuda()

model_path = r"C:\projects\LinearEstimators\best_vgg.pth"
checkpoint = torch.load(model_path)['state_dict']
model.load_state_dict(checkpoint)

activation = {}
def get_activation(name):
	def hook(model, input, output):		
		new_outputs = output.detach().view(BATCH_SIZE, -1).cpu()
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

layers = [module for module in model.features.modules() if type(module) != nn.Sequential][1:]
ctr = 0 
for k, layer in enumerate(layers):
	if type(layer) == torch.nn.modules.activation.ReLU:
		ctr += 1

args = get_args()
Y = torch.cat([target for (data, target) in tqdm(data_loader)]).detach()

for k, layer in enumerate(layers):
	if k <= 15 :
		continue
	print(f"LAYER {k}")
	if type(layer) == torch.nn.modules.activation.ReLU:	
		layer_name = f'layer_{k}'
		handle = layer.register_forward_hook(get_activation(layer_name))
		for i, (input, target) in tqdm(enumerate(data_loader), total=len(data_loader)):	
			output = model(input)

		X = activation[list(activation.keys())[0]]
		start = time.time()
		model = train_model(X, Y, method='WF', trees=args.trees,
            depth=args.depth, features='auto', state=2000, \
            nnormalization='volume')

		alpha, n_wavelets, errors = model.evaluate_smoothness(m=args.num_wavelets)
		print(f"time:{time.time()-start}, alpha  is {alpha}")		
		# seperability.get_seperability(model, 250, X=activation)
		handle.remove()
		del activation[layer_name]

