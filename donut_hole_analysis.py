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
	parser.add_argument('--folder_path', type=str)
	args, unknown = parser.parse_known_args()
	return args

def main(args):
	json_file_paths = glob(os.path.join(args.folder_path, "*.json"))

	for json_path in json_file_paths:
		with open(json_path, "r+") as f:
			result = json.load(f)		
		donut_hole_size = json_path.split('\\')[-1].split('_')[-1].replace('.json', '')
		points = result["points"]
		alphas = np.array(result["alphas"])
		plt.plot(points, alphas, label = donut_hole_size, \
			marker='o', markerfacecolor='black', markersize=12, linewidth=4)		

	plt.xlabel(f'dataset size')
	plt.ylabel(f'evaluate_smoothnes index- alpha')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	args = get_args()
	main(args)
