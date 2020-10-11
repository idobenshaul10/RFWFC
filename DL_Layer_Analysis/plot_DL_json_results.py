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
# USAGE: python .\DL_Layer_Analysis\plot_DL_json_results.py --main_dir C:\projects\RFWFC\results\mnist

def get_args():
	parser = argparse.ArgumentParser(description='Network Smoothness Script')	
	parser.add_argument('--main_dir', type=str ,help='Results folder', default=None)
	parser.add_argument('--checkpoints','--list', nargs='+', default=None)
	args = parser.parse_args()
	return args

def plot_epochs(main_dir, checkpoints=None, plot_test=True, add_fill=False, remove_layers=0):
	if plot_test:
		fig, axes = plt.subplots(1, 2)
		axes[0].set_title("Alphas for different ARCHS - 5-FOLD")
		axes[1].set_title("Test scores different ARHCS - 5-FOLD")
		# axes[2].set_title("Clustering Metrics")
	else:
		fig, axes = plt.subplots(1, 1)
		axes = [axes]

	if checkpoints is not None:
		file_paths = checkpoints		
	else:
		file_paths = list(Path(main_dir).glob('**/*.json'))
		file_paths = [str(k) for k in file_paths]
		file_paths.sort(key=lambda x: int(x.split('\\')[-2].split('.')[-2]))
	clustering_stats = None
	# element = 2 if main_dir is not None else 1


	colors = ['chocolate', 'dodgerblue', 'darkgreen', 'orchid']
	# labels = ['FMNIST BAD', 'FMNIST NORMAL', 'FMNIST RESIDUAL', 'new']
	labels = ['CIFAR10 BAD', 'CIFAR10 NORMAL', 'CIFAR10 RESIDUAL']
	for idx, file_path in enumerate(file_paths):
		file_path = str(file_path)
		epoch = file_path.split('\\')[-2].split('.')[-2]
		# eps = file_path.split('\\')[-element].split('.')[1]
		
		with open(file_path, "r+") as f:			
			result = json.load(f)		
		
		sizes = result["sizes"]
		alphas = result["alphas"]

		if remove_layers > 0:
			sizes, alphas = sizes[:-remove_layers], alphas[:-remove_layers]

		test_stats = None
		if 'test_stats' in result:
			test_stats = result['test_stats']
		if 'clustering_stats' in result:
			clustering_stats = result['clustering_stats']
		if add_fill:	
			axes[0].fill_between(sizes, [k[0] for k in alphas], [k[-1] for k in alphas], \
				alpha=0.2, linewidth=4)		

		axes[0].plot(sizes, [np.array(k).mean()	 for k in alphas], label=labels[idx], color=colors[idx])
		if test_stats is not None and plot_test:
			axes[1].scatter(epoch, [test_stats['top_1_accuracy']], label=labels[idx], color=colors[idx])

		lines = ["-","--","-.",":"]
		linecycler = cycle(lines)

		if clustering_stats is not None and plot_test:
			keys = sorted(list(clustering_stats.keys()))
			if len(keys) == 0:
				continue
			stat_names = clustering_stats[list(keys)[0]].keys()			
			for chosen_stat in stat_names:
				if chosen_stat != 'silhouette_score':
					continue
				values = [clustering_stats[k][chosen_stat] for k in keys]
				axes[2].plot(keys, values, next(linecycler), color=colors[idx], label=f"{chosen_stat}")

	plt.legend()
	plt.xlabel(f'Epoch')
	plt.ylabel(f'evaluate_smoothnes index- alpha')
	plt.show()

if __name__ == '__main__':
	args = get_args()	
	plot_epochs(args.main_dir, args.checkpoints, plot_test=True, add_fill=False)