import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter, defaultdict
import time
import json
import argparse

def get_args():
	parser = argparse.ArgumentParser(description='analysis of json files for alpha score')		
	parser.add_argument('--json_file_path', type=str)
	args, unknown = parser.parse_known_args()
	return args

def main(args):
	K = 5
	with open(args.json_file_path, "r+") as f:
		result = json.load(f)
	points = result["points"][-K:]
	alphas = np.array(result["alphas"][-K:])
	mean_alpha = alphas.mean()
	N = result["flags"]["dimension"]
	estimated_desired_value = 1/(2*(N-1))
	absolute_error = abs(mean_alpha -estimated_desired_value)
	relative_error = absolute_error/abs(estimated_desired_value)
	print(f"mean_alpha:{mean_alpha}, estimated_desired_value:{estimated_desired_value}, "
		+ f"absolute_error:{absolute_error}, relative_error:{relative_error}")

if __name__ == '__main__':
	args = get_args()
	main(args)
