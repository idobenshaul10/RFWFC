import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter, defaultdict
import time
import json
import argparse
from glob import glob

def get_args():
	parser = argparse.ArgumentParser(description='analysis of json files for alpha score')		
	parser.add_argument('--folder_path', required=False, type=str)
	parser.add_argument('--per_donut_path', required=False, type=str)
	args, unknown = parser.parse_known_args()
	return args

def compare_on_folder(args):
	json_file_paths = glob(os.path.join(args.folder_path, "*.json"))
	json_file_paths = [k for k in json_file_paths if "DONUT_" not in k]

	for json_path in json_file_paths:
		with open(json_path, "r+") as f:
			result = json.load(f)		
		donut_hole_size = json_path.split('\\')[-1].split('_')[-1].replace('.json', '')
		points = np.array(result["points"])
		alphas = np.array(result["alphas"])

		indices = np.array(points) > 2000
		points = points[indices]
		alphas = alphas[indices]
		plt.plot(points, alphas, label = donut_hole_size, \
			marker='o', markerfacecolor='black', markersize=2, linewidth=4)		

	plt.figure(1)
	plt.xlabel(f'dataset size')
	plt.ylabel(f'smoothness index- alpha')
	plt.title("Alpha smoothness for different Donut-Hole Diameters")

	current_handles, current_labels = plt.gca().get_legend_handles_labels()	
	indices = np.argsort(current_labels)[::-1]
	reversed_handles = list(np.array(current_handles)[indices])
	reversed_labels = list(np.array(current_labels)[indices])
	plt.legend(reversed_handles,reversed_labels, prop={"size":5})	
	
	plt.savefig(os.path.join(args.folder_path, 'plot.png'), \
		dpi=300, bbox_inches='tight')
	plt.figure(2)

def draw_per_donut_size(args):
	json_path = args.per_donut_path
	with open(json_path, "r+") as f:
		result = json.load(f)
	points = result["points"]
	alphas = np.array(result["alphas"])
	plt.scatter(points, alphas, label = "donut_hole_size", marker='o')

	plt.figure(1)
	plt.title(json_path.split('\\')[-1][:-5])
	plt.xlabel(f'donut hole diameter')
	plt.ylabel(f'smoothness index- alpha')
	plt.legend()

	plt.savefig(os.path.join(os.path.dirname(json_path), 'donuthole_plot.png'), \
		dpi=300, bbox_inches='tight')
	plt.figure(2)

if __name__ == '__main__':
	args = get_args()
	compare_on_folder(args)
	# draw_per_donut_size(args)