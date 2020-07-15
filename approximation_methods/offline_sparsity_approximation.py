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

def plot_derivatives(taus, alphas, total_sparsities, output_path):

	save = False
	tau_save_path = os.path.join(output_path, "tau_sparsity_derivatives.png")
	alpha_save_path = os.path.join(output_path, "alpha_sparsity_derivatives.png")
	h = (alphas[1] - alphas[0])
	# first order
	diffs = np.array(total_sparsities[1:]) - np.array(total_sparsities[:-1])
	diffs = diffs
	
	plt.figure(3)
	plt.title("tau vs. sparsity derivative")
	plt.xlabel('tau')
	plt.ylabel('sparsity derivative')
	plt.plot(taus[1:], diffs)
	plt.pause(1)
	
	if save:
		plt.savefig(tau_save_path, dpi=300, bbox_inches='tight')

	# plt.figure(4)
	# plt.title("alpha vs. sparsity derivative")
	# plt.xlabel('alpha')
	# plt.ylabel('sparsity derivative')
	# plt.plot(alphas[1:], -diffs)
	# plt.pause(1)
	# if save:
	# 	plt.savefig(alpha_save_path, dpi=300, bbox_inches='tight')

def find_best_fit_alpha(errors_data, output_path, verbose=True, p=2):
	m_step, start_m_step = 10, 10
	norms = np.array(errors_data["norms"])
	tau_save_path = os.path.join(output_path, "tau_sparsity.png")
	alpha_save_path = os.path.join(output_path, "alpha_sparsity.png")

	taus = np.arange(0.8, 1.6, 0.01)
	alphas = ((1/taus) - 1/p)
	save = True	
	total_sparsities, total_alphas = [], []
	
	for tau in taus:			
		tau_sparsity = np.power(np.power(norms, tau).sum(), (1/tau))
		total_sparsities.append(tau_sparsity)
		print(f"tau:{tau}, tau_sparsity:{tau_sparsity}")

	if verbose:
		plt.figure(1)
		plt.title("tau vs. sparsity")
		plt.xlabel('tau')
		plt.ylabel('sparsity')
		plt.plot(taus, total_sparsities)
		plt.pause(10)
		
		if save:
			plt.savefig(tau_save_path, dpi=300, bbox_inches='tight')

		plt.figure(2)
		plt.title("alpha vs. sparsity")
		plt.xlabel('alpha')
		plt.ylabel('sparsity')
		plt.plot(alphas, total_sparsities)
		plt.pause(1)

		if save:
			plt.savefig(alpha_save_path, dpi=300, bbox_inches='tight')

	# plot_derivatives(taus, alphas, total_sparsities, output_path)

		

if __name__ == "__main__":
	json_path = sys.argv[1]
	output_path = None
	if len(sys.argv) > 2:
		output_path = sys.argv[2]
	else:		
		output_path = os.path.dirname(sys.argv[1])
	f = open(json_path)
	errors_data = json.load(f)
	find_best_fit_alpha(errors_data, output_path, verbose=True)

