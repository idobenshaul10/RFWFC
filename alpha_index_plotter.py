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
from utils import normalize_data, kfold_alpha_smoothness
import logging
ion()

def plot_dataset(X, Y):
	colors = ["red" if y==1 else "blue" for y in Y]
	groups = ("in ball", "outside ball")
	plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.5)
	plt.title(f'Dataset:size is {X.shape[0]}')
	plt.xlabel('x')
	plt.ylabel('y')
	# plt.savefig(os.path.join(folder_path, 'dataset_first_two_dims.png'), \
	# 	dpi=300, bbox_inches='tight')
	plt.show()

def plot_alpha_per_tree_number(flags, data_str, output_path):
	tree_sizes ,alphas = [], []
	pointGen = PointGenerator(dim=flags.dimension, seed=flags.seed)
	# samples =[500, 750, 1000, 1500, 2000, 2500] #+ list(range(1550, 1550, 200))

	dataset_size = 10000
	depth = flags.depth
	X, y = pointGen[dataset_size]
	logging.info(f"LABELS COUNTER: {Counter(y)}")

	for n_trees in tqdm(range(100, 150, 10)):		
		# print(f"\t\t1:{y.sum()}, 0:{dataset_size-y.sum()}")
		regressor = WaveletsForestRegressor(\
			regressor=flags.regressor, trees=n_trees, \
			features=flags.features, seed=flags.seed, \
			depth=depth)

		rf = regressor.fit(X, y)		
		# if flags.dimension == 2 :
		# 	rf.visualize_classifier()
		alpha, n_wavelets, errors = rf.evaluate_smoothness(m=1000)		

		alphas.append(alpha)
		tree_sizes.append(n_trees)
		del regressor

	plt.figure(1)
	plt.plot(tree_sizes, alphas)	
	# draw_predictive_line(flags.dimension, p=2)

	plt.title(data_str + f" dataset_size: {dataset_size}")
	plt.xlabel(f'# of trees')
	plt.ylabel(f'evaluate_smoothnessothness index- alpha')
	
	plt.show(block=True)


def plot_alpha_per_depth(flags, data_str, output_path):
	depths ,alphas = [], []
	pointGen = PointGenerator(dim=flags.dimension, seed=flags.seed)
	# samples =[500, 750, 1000, 1500, 2000, 2500] #+ list(range(1550, 1550, 200))

	dataset_size = 2400
	X, y = pointGen[dataset_size]

	for depth in tqdm(range(5, 100, 5)):		
		print(f"\t\t1:{y.sum()}, 0:{dataset_size-y.sum()}")
		try:
			del regressor
			print("deleted regressor")
		except:
			pass
		regressor = WaveletsForestRegressor(\
			regressor=flags.regressor, trees=flags.trees, \
			features=flags.features, seed=flags.seed, \
			depth=depth)

		rf = regressor.fit(X, y)
		if flags.dimension == 2 :
			rf.visualize_classifier()
		alpha, n_wavelets, errors = rf.evaluate_smoothness(m=10000)		

		alphas.append(alpha)
		depths.append(depth)

	plt.figure(1)
	plt.plot(depths, alphas)	
	# draw_predictive_line(flags.dimension, p=2)

	plt.title(data_str)
	plt.xlabel(f'depths')
	plt.ylabel(f'evaluate_smoothnessothness index- alpha')
	
	plt.show(block=True)

# MIN_SIZE = 5000000
# MAX_SIZE = 10000002
# STEP = 1000000

MIN_SIZE = 1000
MAX_SIZE = 200000
STEP = 5000



def plot_alpha_per_num_sample_points(flags, \
	data_str, normalize=True, output_path=None):
	sizes ,alphas, stds = [], [], []
	pointGen = PointGenerator(dim=flags.dimension, seed=flags.seed)	
	seed_dict = defaultdict(list)

	for dataset_size in tqdm(range(MIN_SIZE, MAX_SIZE, STEP)):
		x, y = pointGen[dataset_size]
		logging.info(f"LABELS COUNTER: {Counter(y.squeeze())}")
		# rf.visualize_estimator("depth_100")

		# if flags.dimension == 2:
		# 	rf.visualize_classifier()
		mean_alpha, std_alpha, num_wavelets, norm_m_term = \
				kfold_alpha_smoothness(x, y, t_method=flags.regressor, \
				num_wavelets=1000, n_folds=10, n_trees=flags.trees, m_depth=flags.depth,
                n_features='auto', n_state=2000, normalize=True, norm_normalization='samples')

		# alpha, n_wavelets, errors = rf.evaluate_smoothness(m=10000)
		stds.append(std_alpha)
		alphas.append(mean_alpha)
		# import pdb; pdb.set_trace()
		
		if True:
			sizes.append(dataset_size)	

	print(f'stds:{stds}')
	plt.figure(1)
	plt.plot(sizes, alphas)
	# plt.plot(sizes, results)
	draw_predictive_line(flags.dimension, p=2)

	plt.title(data_str)
	plt.xlabel(f'dataset size')
	plt.ylabel(f'evaluate_smoothnes index- alpha')

	save_graph=False
	if save_graph:
		save_path = os.path.join(\
			output_path, f"{str(flags.dimension)}.png")
		# import pd b; pdb.set_trace()
		plt.savefig(save_path, \
			dpi=300, bbox_inches='tight')
	plt.show(block=True)

def draw_predictive_line(n, p=2):
	x = np.linspace(MIN_SIZE,MAX_SIZE,100)
	y = [1/(p*(n-1)) for k in x]
	plt.plot(x, y, '-r', label='y=2x+1')
	# plt.grid()

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



