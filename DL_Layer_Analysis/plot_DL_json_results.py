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
# USAGE: python .\DL_Layer_Analysis\plot_DL_json_results.py --main_dir C:\projects\RFWFC\results\mnist

def get_args():
	parser = argparse.ArgumentParser(description='Network Smoothness Script')	
	parser.add_argument('--main_dir', type=str ,help='Results folder')
	args = parser.parse_args()
	return args

def plot_epochs(main_dir, plot_test=True, add_fill=False):
	if plot_test:
		fig, axes = plt.subplots(1, 2)
		axes[0].set_title("alphas for different epochs")
		axes[1].set_title("Test scores different epochs")
	else:
		fig, axes = plt.subplots(1, 1)
		axes = [axes]

	file_paths = list(Path(main_dir).glob('**/*.json'))
	file_paths = [str(k) for k in file_paths]	
	file_paths.sort(key=lambda x: int(x.split('\\')[-2].split('.')[-2]))	
	# plt.title("alphas for different epochs")
	
	for file_path in file_paths:
		file_path = str(file_path)	
		epoch = file_path.split('\\')[-2].split('.')[-2]
		eps = file_path.split('\\')[-2].split('.')[1]	
		with open(file_path, "r+") as f:
			result = json.load(f)	
		sizes = result["sizes"]
		alphas = result["alphas"]

		test_stats = None
		if 'test_stats' in result:
			test_stats = result['test_stats']
		if add_fill:	
			axes[0].fill_between(sizes, [k[0] for k in alphas], [k[1] for k in alphas], \
				alpha=0.2, linewidth=4)
		axes[0].plot(sizes, [np.array(k).mean()	 for k in alphas], label=f"{epoch}")
		if test_stats is not None and plot_test:
			axes[1].scatter(epoch, [test_stats['top_1_accuracy']], label=f"{epoch}")

	plt.legend()
	plt.xlabel(f'layer')
	plt.ylabel(f'evaluate_smoothnes index- alpha')
	plt.show()

if __name__ == '__main__':
	args = get_args()	
	plot_epochs(args.main_dir, add_fill=False)