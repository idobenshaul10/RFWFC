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
from utils import normalize_data, run_alpha_smoothness, kfold_alpha_smoothness, \
	kfold_regression_mse, train_model

from alpha_index_plotter import plot_dataset
import logging	
import json
import time

# from sklearn.tree.export import export_text
# def feature_rules(model, feature_names=['x','y']):
# 	tree_rules = export_text(model, feature_names=feature_names)
# 	import pdb; pdb.set_trace()

from sklearn.tree import _tree
def tree_to_code(tree, feature_names, data):
	output_str = ""
	tree_ = tree.tree_
	feature_name = [
		feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
		for i in tree_.feature
	]
	print("def tree({}):".format(", ".join(feature_names)))

	def recurse(node, depth):
		indent = "  " * depth
		if tree_.feature[node] != _tree.TREE_UNDEFINED:
			name = feature_name[node]
			threshold = tree_.threshold[node]
			print("{}if {} <= {}:".format(indent, name, threshold))
			recurse(tree_.children_left[node], depth + 1)
			print("{}else:  # if {} > {}".format(indent, name, threshold))
			recurse(tree_.children_right[node], depth + 1)
		else:
			print("{}return {}".format(indent, tree_.value[node]))
	recurse(0, 1)

def plot_intersection_volumes(flags, data_str, dataset_size=10000, normalize=True):
	folder_path = r"C:\projects\RFWFC\results\intersection_size"	
	add_noisy_channels = False
	start = time.time()
	sizes ,alphas, stds = [], [], []
	pointGen = PointGenerator(dim=flags.dimension, seed=flags.seed, \
		add_noisy_channels=add_noisy_channels, donut_distance=flags.donut_distance)
	cube_length = pointGen.cube_length
	seed_dict = defaultdict(list)
	N_wavelets = flags.num_wavelets
	donut_distance = flags.donut_distance
	norm_normalization = 'volume'
	normalize = True
	model = None

	X, y = pointGen[dataset_size]
	# plot_dataset(X ,y, donut_distance)	
	model = train_model(X, y, method=flags.regressor, mode='regression', trees=flags.trees,
		depth=flags.depth, nnormalization=norm_normalization, cube_length=cube_length)

	# model = train_model(X, y, method=flags.regressor, mode='classification', trees=flags.trees,
		# depth=flags.depth, nnormalization=norm_normalization)


	model.visualize_classifier()	

	exit()
	
	# plt.figure(1)
	# plt.clf()
	# plt.plot(sizes, alphas)
	
	print_data_str = data_str.replace(':', '_').replace(' ', '').replace(',', '_')	
	file_name = f"STEP_{STEP}_MIN_{MIN_SIZE}_MAX_{MAX_SIZE}_{print_data_str}_Wavelets_{N_wavelets}_Norm_{norm_normalization}_IsNormalize_{normalize}_noisy_{add_noisy_channels}_"+ \
		f"donut_distance_{donut_distance}"
	dir_path = os.path.join(output_path, 'decision_tree_with_bagging', str(flags.dimension))
	
	if not os.path.isdir(dir_path):
		os.mkdir(dir_path)
	img_file_name =file_name + ".png"	

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



