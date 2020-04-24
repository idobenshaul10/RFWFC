import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sphere_loader import PointGenerator
from collections import Counter, defaultdict
from mpl_toolkits.mplot3d import Axes3D
from random_forest import WaveletsForestRegressor
from matplotlib.pyplot import plot, ion, show
from utils import normalize_data, run_alpha_smoothness, kfold_alpha_smoothness
import logging	
import time
import json
ion()

def plot_dataset(X, Y):
	colors = ["red" if y==1 else "blue" for y in Y]
	groups = ("in ball", "outside ball")
	plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.5)
	plt.title(f'Dataset:size is {X.shape[0]}')
	plt.xlabel('x')
	plt.ylabel('y')	
	plt.draw()
	plt.pause(0.001)

# samples
MIN_SIZE = 10000
MAX_SIZE = 100001
STEP = 10000

# MIN_SIZE = 1
# MAX_SIZE = 80
# STEP = 10

def plot_alpha_per_num_sample_points(flags, \
	data_str, normalize=True, output_path=''):
	n_folds = 5
	add_noisy_channels = False
	start = time.time()
	sizes ,alphas, stds = [], [], []
	pointGen = PointGenerator(dim=flags.dimension, seed=flags.seed, \
		add_noisy_channels=add_noisy_channels, donut_distance=flags.donut_distance)	
	seed_dict = defaultdict(list)
	N_wavelets = flags.num_wavelets
	donut_distance = flags.donut_distance
	norm_normalization = 'volume'
	normalize = True
	if not os.path.isdir(output_path):
		os.mkdir(output_path)

	for dataset_size in tqdm(range(MIN_SIZE, MAX_SIZE, STEP)):
		x, y = pointGen[dataset_size]
		# plot_dataset(x,y)
		logging.info(f"LABELS COUNTER: {Counter(y.squeeze())}")		

		mean_alpha, std_alpha, num_wavelets, norm_m_term = \
			run_alpha_smoothness(x, y, t_method=flags.regressor, \
				num_wavelets=N_wavelets, n_folds=n_folds, n_trees=flags.trees, m_depth=flags.depth,
				n_features='auto', n_state=2000, normalize=normalize, norm_normalization=norm_normalization)
		
		stds.append(std_alpha)
		alphas.append(mean_alpha)
		
		if True:
			sizes.append(dataset_size)	

	print(f'stds:{stds}')
	plt.clf()
	plt.figure(1)
	plt.plot(sizes, alphas)	
	
	write_data = {}
	write_data['points'] = sizes
	write_data['alphas'] = alphas	
	write_data['flags'] = vars(flags)
	write_data['MIN_SIZE'] = MIN_SIZE
	write_data['MAX_SIZE'] = MAX_SIZE
	write_data['STEP'] = STEP
	write_data['N_wavelets'] = N_wavelets
	write_data['norm_normalization'] = norm_normalization
	write_data['normalize'] = normalize
	write_data['add_noisy_channels'] = add_noisy_channels
	write_data['donut_distance'] = donut_distance

	desired_value = draw_predictive_line(flags.dimension, p=2)
	last_alpha = alphas[-1]
	print_data_str = data_str.replace(':', '_').replace(' ', '').replace(',', '_')	
	file_name = f"STEP_{STEP}_MIN_{MIN_SIZE}_MAX_{MAX_SIZE}_{print_data_str}_Wavelets_{N_wavelets}_Norm_{norm_normalization}_IsNormalize_{normalize}_noisy_{add_noisy_channels}_"+ \
		f"donut_distance_{donut_distance}"
	dir_path = os.path.join(output_path, 'decision_tree_with_bagging', str(flags.dimension))
	
	if not os.path.isdir(dir_path):
		os.mkdir(dir_path)
	img_file_name =file_name + ".png"
	json_file_name = file_name + ".json"
	
	with open(os.path.join(dir_path, json_file_name), "w+") as f:
		json.dump(write_data, f)

	plt.title(data_str)
	plt.xlabel(f'dataset size')
	plt.ylabel(f'evaluate_smoothnes index- alpha')

	save_graph=True
	if save_graph:
		
		if not os.path.isdir(dir_path):
			os.mkdir(dir_path)
		
		save_path = os.path.join(dir_path, img_file_name)
		print(f"save_path:{save_path}")
		plt.savefig(save_path, \
			dpi=300, bbox_inches='tight')
	end = time.time()
	print(f"total time is {end-start}")
	plt.show(block=False)
def plot_alpha_per_depth(flags, \
	data_str, normalize=True, output_path=''):
	n_folds = 5
	add_noisy_channels = False
	start = time.time()
	sizes ,alphas, stds = [], [], []
	pointGen = PointGenerator(dim=flags.dimension, seed=flags.seed, \
		add_noisy_channels=add_noisy_channels)	
	seed_dict = defaultdict(list)
	N_wavelets = flags.num_wavelets
	norm_normalization = 'volume'
	normalize = True
	if not os.path.isdir(output_path):
		os.mkdir(output_path)

	dataset_size = 30000
	x, y = pointGen[dataset_size]
	logging.info(f"LABELS COUNTER: {Counter(y.squeeze())}")
	for depth in tqdm(range(1, 20)):
		kfold_alpha_smoothness(x, y, t_method=flags.regressor, \
			num_wavelets=N_wavelets, n_folds=n_folds, n_trees=1, m_depth=depth,
			n_features='auto', n_state=2000, normalize=normalize, norm_normalization=norm_normalization)		

	print(f'stds:{stds}')
	plt.figure(1)
	plt.plot(sizes, alphas)	
def draw_predictive_line(n, p=2):	
	x = np.linspace(MIN_SIZE, list(range(MIN_SIZE,MAX_SIZE,STEP))[-1],100)
	desired_value = 1/(p*(n-1))
	y = [desired_value for k in x]
	plt.plot(x, y, '-r', label='y=2x+1')
	return desired_value

def save_results(sizes, alphas, X, y, data_str, output_path):
	folder_path = os.path.join(output_path, data_str)
	if not os.path.isdir(folder_path):
		os.mkdir(folder_path)

	plt.figure(1)
	plt.plot(sizes, alphas)
	plt.title(data_str)
	plt.xlabel(f'dataset size')
	plt.ylabel(f'smoothness index- alpha')

	plt.savefig(os.path.join(folder_path, 'alpha.png'), \
		dpi=300, bbox_inches='tight')	

	plt.figure(2)
	plot_dataset(X, y, folder_path)
def plot_alpha_per_tree_number(flags, data_str, output_path):
	tree_sizes ,alphas, stds = [], [], []
	pointGen = PointGenerator(dim=flags.dimension, seed=flags.seed)	
	n_folds = 5

	dataset_size = 10000
	depth = flags.depth
	X, y = pointGen[dataset_size]	

	seed_dict = defaultdict(list)
	N_wavelets = flags.num_wavelets
	norm_normalization = 'volume'
	normalize = True
	if not os.path.isdir(output_path):
		os.mkdir(output_path)

	for n_trees in tqdm(range(MIN_SIZE, MAX_SIZE, STEP)):		
		mean_alpha, std_alpha, num_wavelets, norm_m_term = \
			kfold_alpha_smoothness(X, y, t_method=flags.regressor, \
				num_wavelets=N_wavelets, n_folds=n_folds, n_trees=n_trees, m_depth=flags.depth,
				n_features='auto', n_state=2000, normalize=normalize, norm_normalization=norm_normalization)		

		stds.append(std_alpha)
		alphas.append(mean_alpha)
		tree_sizes.append(n_trees)
	
	plt.figure(1)
	plt.plot(tree_sizes, alphas)	
	draw_predictive_line(flags.dimension, p=2)

	plt.title(data_str + f" dataset_size: {dataset_size}")
	plt.xlabel(f'# of trees')
	plt.ylabel(f'evaluate_smoothness index- alpha')	
	plt.show(block=True)

	write_data = {}
	write_data['points'] = tree_sizes
	write_data['alphas'] = alphas
	write_data['flags'] = vars(flags)
	write_data['MIN_SIZE'] = MIN_SIZE
	write_data['MAX_SIZE'] = MAX_SIZE
	write_data['STEP'] = STEP
	write_data['N_wavelets'] = N_wavelets
	write_data['norm_normalization'] = norm_normalization
	write_data['normalize'] = normalize	

	desired_value = draw_predictive_line(flags.dimension, p=2)
	last_alpha = alphas[-1]
	print_data_str = data_str.replace(':', '_').replace(' ', '').replace(',', '_')	
	file_name = f"STEP_{STEP}_MIN_{MIN_SIZE}_MAX_{MAX_SIZE}_{print_data_str}_Wavelets_{N_wavelets}_Norm_{norm_normalization}_IsNormalize_{normalize}"
	dir_path = os.path.join(output_path, 'decision_tree_with_bagging', str(flags.dimension))
	
	if not os.path.isdir(dir_path):
		os.mkdir(dir_path)
	img_file_name =file_name + ".png"
	json_file_name = file_name + ".json"
	
	with open(os.path.join(dir_path, json_file_name), "w+") as f:
		json.dump(write_data, f)

	plt.title(data_str)
	plt.xlabel(f'dataset size')
	plt.ylabel(f'evaluate_smoothnes index- alpha')

	save_graph=True
	if save_graph:		
		if not os.path.isdir(dir_path):
			os.mkdir(dir_path)
		
		save_path = os.path.join(dir_path, img_file_name)
		print(f"save_path:{save_path}")
		plt.savefig(save_path, \
			dpi=300, bbox_inches='tight')
	end = time.time()
	print(f"total time is {end-start}")
	plt.show(block=True)