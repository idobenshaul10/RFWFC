from __future__ import print_function
import os 
import sys
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, ion, show
import numpy as np
import importlib
import os,sys,inspect
from pathlib import Path
import json
from itertools import cycle
from scipy.stats import pearsonr as pearson
import seaborn as sns
from collections import defaultdict
# USAGE: python .\DL_Layer_Analysis\plot_clustering_correlations.py --main_dir C:\projects\RFWFC\results\mnist
# USAGE: python .\DL_Layer_Analysis\plot_clustering_correlations.py --checkpoints C:\projects\DL_Smoothness_Results\trained_models\TWO_LAYER_RESIDUAL\cifar10\cifar10_2020_10_15-10_47_30_PM\DL_Analysis\cifar10_1_15_0.1_0.40\result.json C:\projects\DL_Smoothness_Results\trained_models\TWO_LAYER_RESIDUAL\mnist\mnist_2020_10_15-07_59_36_PM\DL_Analysis\mnist_1_15_0.1_0.40\result.json C:\projects\DL_Smoothness_Results\trained_models\TWO_LAYER_RESIDUAL\fashion_mnist\fashion_mnist_2020_10_15-09_32_15_PM\DL_Analysis\fashion_mnist_1_15_0.1_0.40\result.json

def get_args():
	parser = argparse.ArgumentParser(description='Network Smoothness Script')	
	parser.add_argument('--main_dir', type=str ,help='Results folder', default=None)
	parser.add_argument('--checkpoints','--list', nargs='+', default=None)
	args = parser.parse_args()
	return args

def plot_epochs(main_dir, checkpoints=None, plot_test=True, add_fill=False, remove_layers=0):	
	if checkpoints is not None:
		file_paths = checkpoints		
	else:
		file_paths = list(Path(main_dir).glob('**/*.json'))
		file_paths = [str(k) for k in file_paths]
		file_paths.sort(key=lambda x: int(x.split('\\')[-2].split('.')[-2]))	
	clustering_stats = None

	fig, ax = plt.subplots(figsize=(8,6))
	ax.set_ylim([0,3])
	
	use_pearson = True
	correlations = defaultdict(list)
	for idx, file_path in enumerate(file_paths):
		file_path = str(file_path)
		epoch = file_path.split('\\')[-2].split('.')[-2]		
		
		with open(file_path, "r+") as f:
			result = json.load(f)		
		
		sizes = result["sizes"]
		alphas = result["alphas"]

		if remove_layers > 0:
			sizes, alphas = sizes[:-remove_layers], alphas[:-remove_layers]
		
		if 'clustering_stats' in result:
			clustering_stats = result['clustering_stats']

		alphas = np.array(alphas).mean(axis=1)		
		
		if clustering_stats is not None and plot_test:
			keys = sorted(list(clustering_stats.keys()))
			if len(keys) == 0:
				continue
			stat_names = clustering_stats[list(keys)[0]].keys()			
			for chosen_stat in stat_names:				
				values = [clustering_stats[k][chosen_stat] for k in keys]
				if use_pearson:
					correlations[chosen_stat].append(pearson(alphas, values)[0])
				else:
					a = np.corrcoef(alphas, values)
					correlations[chosen_stat].append(a)
		
	
	metrics = list(correlations.keys())
	values = np.array([correlations[k] for k in metrics]).reshape(6, 3)
	dataset_names = ["BAD", "GOOD", "RESIDUAL"]

	sns.heatmap(values,  annot=True, vmin=0., vmax=1.,\
		xticklabels=dataset_names, yticklabels=metrics)
	print(correlations)

	plt.title("Pearson Correlation: Besov-Smoothness vs. Clustering")	
	plt.show()

if __name__ == '__main__':
	args = get_args()	
	plot_epochs(args.main_dir, args.checkpoints, plot_test=True, add_fill=False)