import os 
import sys

# import cv2
import torch
import numpy as np
import logging
from sklearn import tree, linear_model, ensemble
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix, lil_matrix
import operator
import code
from functools import reduce
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.pyplot import plot, ion, show
import random
from sklearn import metrics
import math
from sklearn.metrics import *
import json
from tqdm import tqdm

def save_graphs(output_path, steps, alphas, consts):
	if not os.path.isdir(output_path):
		os.mkdir(output_path)
	alphas_path = os.path.join(output_path, "alpha.png")
	consts_path = os.path.join(output_path, "const.png")

	plt.clf()
	plt.title("alphas vs. num wavelets")
	plt.plot(steps, alphas)
	plt.xlabel('#wavelets')
	plt.ylabel('alphas')
	plt.savefig(alphas_path, dpi=300, bbox_inches='tight')

	plt.clf()
	plt.plot(steps, consts)
	plt.title("const vs. num wavelets")
	plt.xlabel('#wavelets')
	plt.ylabel('consts')
	plt.savefig(consts_path, dpi=300, bbox_inches='tight')

def plot_total_alphas(n_wavelets, total_alpha_consts, alphas, output_path):
	save_path = os.path.join(output_path, "total_const.png")
	plt.figure(3)
	plt.title("Consts vs. Num Wavelets")
	plt.xlabel('Num Wavelets')
	plt.ylabel('Consts')	
	for i in range(0, len(total_alpha_consts), 10):
		plt.plot(n_wavelets, total_alpha_consts[i], label=f"alpha:{alphas[i]:.2f}")		
	plt.legend()
	plt.pause(2)
	plt.savefig(save_path, dpi=300, bbox_inches='tight')


def find_best_fit_alpha(errors_data, output_path, verbose=True):
	m_step, start_m_step = 10, 10
	errors = errors_data["errors"]
	n_wavelets = errors_data["n_wavelets"]
	n_wavelets_log = np.log(np.reshape(n_wavelets, (-1, 1)))
	errors_log = np.log(np.reshape(errors, (-1, 1)))

	mean_path = os.path.join(output_path, "mean_const.png")
	std_path = os.path.join(output_path, "std_const.png")

	print(f"number of errors:{len(errors)}")
	alphas = np.arange(0.05, 1., 0.01)
	total_alpha_consts, total_mean_consts, total_std_consts = [], [], []
	for alpha in alphas:
		a_consts = errors * np.power(n_wavelets, alpha)
		total_alpha_consts.append(a_consts)
		a_const_mean = a_consts.mean()
		a_const_std = a_consts.std()
		total_mean_consts.append(a_const_mean)
		total_std_consts.append(a_const_std)

	if verbose:
		plt.figure(1)
		plt.title("alpha vs. mean const")
		plt.xlabel('alpha')
		plt.ylabel('mean const')
		plt.plot(alphas, total_mean_consts)
		plt.legend()
		plt.pause(1)
		if save:
			plt.savefig(mean_path, dpi=300, bbox_inches='tight')
		# ax.lines.pop(-1)

		plt.figure(2)
		plt.title("wavelet vs. std const")
		plt.xlabel('alpha')
		plt.ylabel('std const')
		plt.plot(alphas, total_std_consts)
		plt.pause(5)
		if save:
			plt.savefig(std_path, dpi=300, bbox_inches='tight')

	plot_total_alphas(n_wavelets, total_alpha_consts, alphas, output_path)

if __name__ == "__main__":
	json_path = sys.argv[1]
	output_path = None
	if len(sys.argv) > 2:
		output_path = sys.argv[2]
	else:		
		output_path = os.path.dirname(sys.argv[1])
	f = open(json_path)
	errors_data = json.load(f)
	find_best_fit_alpha(errors_data, output_path, verbose=False)

