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
from clustering import kmeans_cluster
from utils.utils import *
import time
import json
from multiprocessing import Process
from glob import glob

def get_args():
	parser = argparse.ArgumentParser(description='Network Smoothness Script')	
	parser.add_argument('--trees',default=1,type=int,help='Number of trees in the forest.')	
	parser.add_argument('--depth', default=15, type=int,help='Maximum depth of each tree.Use 0 for unlimited depth.')		
	parser.add_argument('--criterion',default='gini',help='Splitting criterion.')
	parser.add_argument('--bagging',default=0.8,type=float,help='Bagging. Only available when using the "decision_tree_with_bagging" regressor.')
	parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')	
	parser.add_argument('--output_folder', type=str, default=r"C:\projects\results", \
		help='path to save results')
	parser.add_argument('--num_wavelets', default=2000, type=int,help='# wavelets in N-term approx')	
	parser.add_argument('--batch_size', type=int, default=1024)
	parser.add_argument('--env_name', type=str, default="cifar10")
	parser.add_argument('--checkpoints_dir_path', type=str, default="")
	parser.add_argument('--high_range_epsilon', type=float, default=0.1)
	parser.add_argument('--use_clustering', action='store_true', default=False)
	parser.add_argument('--calc_test', action='store_true', default=False)

	args = parser.parse_args()	
	return args

def run_all_comands(args):
	RUN_FREQ = 10
	if not os.path.isdir(args.checkpoints_dir_path):
		raise("Checkpoints directory does not exist!")
	checkpoints = glob(os.path.join(args.checkpoints_dir_path, "*.h5"))	
	checkpoints = [k for k in checkpoints if int(k.split('\\')[-1].split('.')[1]) % RUN_FREQ == 0]
	cmd = get_base_command(args)
	
	for checkpoint_path in tqdm(checkpoints):
		cur_cmd = cmd + f' --checkpoint_path {checkpoint_path}'
		process = Process(target=run_cmd, args=([cur_cmd]))
		process.start()
		process.join()

def run_cmd(cmd):	
	job_output = os.popen(cmd).read()
	job_output = str(job_output)
	print(f"JOB_ID {job_output}", flush=True)

def get_base_command(args):
	cmd = f'python DL_Layer_Analysis/DL_smoothness.py'
	args_dict = vars(args)
	for key, value in args_dict.items():
		if key == 'checkpoints_dir_path':
			continue
		if key == 'use_clustering':
			if value is True:
				cmd += ' --use_clustering'
			continue
		if key == 'calc_test':
			if value is True:
				cmd += ' --calc_test'
			continue				
		value = str(value).replace('&', '\\&')
		cmd += f" --{key} {value}"
	return cmd

if __name__ == '__main__':
	args = get_args()
	run_all_comands(args)