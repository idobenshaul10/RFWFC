import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sphere_experiment.sphere_loader import PointGenerator
from collections import Counter, defaultdict
from mpl_toolkits.mplot3d import Axes3D
from tree_models.random_forest import WaveletsForestRegressor
from matplotlib.pyplot import plot, ion, show
from utils.utils import *
import logging	
import time
import json

ion()
def plot_dataset(X, Y, output_path='', save=False):
	colors = ["red" if y==1 else "blue" for y in Y]
	groups = ("in ball", "outside ball")
	s = [0.5 for n in range(len(X))]
	plt.scatter(X[:, 0], X[:, 1], c=colors, s=s, alpha=0.5)
	plt.title(f'Dataset:size is {X.shape[0]}')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.draw()
	plt.pause(1)
	if save and os.path.isdir(output_path):
		img_file_name = "dataset.png"
		save_path = os.path.join(output_path, img_file_name)
		print(f"save_path:{save_path}")
		plt.savefig(save_path, \
			dpi=300, bbox_inches='tight')

MIN_SIZE = 1000
MAX_SIZE = 35000
STEP = 5000

def plot_alpha_per_num_sample_points(flags, data_str, normalize=True, \
		output_path='', high_range_epsilon=0.1, low_range_epsilon=0.4, power=2, verbose=False):
	
	add_noisy_channels = False
	start = time.time()
	sizes ,alphas, stds = [], [], []
	pointGen = PointGenerator(dim=flags.dimension, seed=flags.seed, \
		add_noisy_channels=add_noisy_channels, donut_distance=flags.donut_distance)
	seed_dict = defaultdict(list)
	N_wavelets = flags.num_wavelets
	donut_distance = flags.donut_distance
	norm_normalization = 'num_samples'
	error_TH = flags.error_TH
	normalize = True
	semi_norm = False
	wavelet_norms = False

	if not os.path.isdir(output_path):
		os.mkdir(output_path)

	for dataset_size in tqdm(range(MIN_SIZE, MAX_SIZE, STEP)):
		x, y = pointGen[dataset_size]
		if verbose:
			plot_dataset(x, y)
		if wavelet_norms:
			save_wavelet_norms(x, y, t_method=flags.regressor, \
				num_wavelets=N_wavelets, m_depth=flags.depth, \
				n_state=2000, normalize=False, \
				norm_normalization=norm_normalization, cube_length=pointGen.cube_length)
			continue

		if semi_norm:
			calculate_besov_semi_norm(x, y, t_method=flags.regressor, \
				num_wavelets=N_wavelets, m_depth=flags.depth, \
				n_state=2000, normalize=False, \
				norm_normalization=norm_normalization, cube_length=pointGen.cube_length)
			continue
		
		mean_alpha, std_alpha, num_wavelets, norm_m_term, model = \
			run_alpha_smoothness(x, y, n_trees=flags.trees, t_method=flags.regressor, \
				num_wavelets=N_wavelets, m_depth=flags.depth, \
				n_state=2000, normalize=False, \
				norm_normalization=norm_normalization, cube_length=pointGen.cube_length, \
				error_TH=flags.error_TH, output_folder=output_path, \
				epsilon_1=high_range_epsilon, epsilon_2=low_range_epsilon, text=str(dataset_size))
		
		stds.append(std_alpha)
		alphas.append(mean_alpha)		
		sizes.append(dataset_size)	

	print(f'stds:{stds}')	
	plt.figure(1)
	plt.clf()
	plt.ylim(0., 1.)
	if type(alphas) == list:
		plt.fill_between(sizes, [k[0] for k in alphas], [k[1] for k in alphas], \
			alpha=0.2, facecolor='#089FFF', \
			linewidth=4)
		plt.plot(sizes, [np.array(k).mean()	 for k in alphas], 'k', color='#1B2ACC')
	else:
		plt.plot(sizes, alphas, 'k', color='#1B2ACC')
	
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
	write_data['error_TH'] = error_TH

	desired_value = draw_predictive_line(flags.dimension, p=power)
	last_alpha = alphas[-1]
	print_data_str = data_str.replace(':', '_').replace(' ', '').replace(',', '_')	
	file_name = f"STEP_{STEP}_MIN_{MIN_SIZE}_MAX_{MAX_SIZE}_{print_data_str}_Wavelets_{N_wavelets}_Norm_{norm_normalization}_IsNormalize_{normalize}_error_TH_{error_TH}"+ \
		f"donut_distance_{donut_distance}"	

	dir_path = output_path
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

def draw_predictive_line(n, p=2):	
	x = np.linspace(MIN_SIZE, list(range(MIN_SIZE,MAX_SIZE,STEP))[-1], STEP)	
	desired_value = 1/(p*(n-1))
	y = [desired_value for k in x]
	plt.plot(x, y, '-r', label='y=2x+1')
	return desired_value