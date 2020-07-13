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
import torch

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

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def plot_derivatives(taus, alphas, total_sparsities, output_path):
	N = 0
	total_sparsities = np.array(total_sparsities)
	# total_sparsities /= 10000
	save = False
	tau_save_path = os.path.join(output_path, "tau_sparsity_derivatives.png")
	alpha_save_path = os.path.join(output_path, "alpha_sparsity_derivatives.png")
	h = 0.01
	# first order
	diffs = np.array(total_sparsities[1:]) - np.array(total_sparsities[:-1])	

	for j in range(N):	
		# index = np.where(abs(taus-1)<1e-6)[0]
		if j != 0:
			diffs = np.array(diffs[1:]) - np.array(diffs[:-1])
		# diffs *= h
		if j != N-1:
			continue
		plt.figure(j)
		# plt.title(f"tau vs. sparsity {j+1}-derivative")
		# plt.xlabel(f'tau')
		# plt.ylabel(f'sparsity {j+1}-derivative')
		# # plt.ylim(-1000, 0.)
		index = np.where(abs(taus[j+1:]-1)<1e-6)[0]
		# plt.plot(taus[j+1:], diffs, zorder=1)
		# plt.scatter([taus[j+1:][index]], [diffs[index]], color="r", zorder=2)

		# plt.title(f"tau vs. sparsity {j+1}-derivative angle")
		# plt.xlabel(f'tau')
		# plt.ylabel(f'sparsity {j+1}-derivative angle')
		# angles = np.rad2deg(np.arctan(diffs))
		# plt.plot(taus[j+1:], angles, zorder=1)
		# plt.scatter([taus[j+1:][index]], [angles[index]], color="r", zorder=2)

		plt.title(f"tau vs. sparsity {j+1}-derivative sigmoid")
		plt.xlabel(f'tau')
		plt.ylabel(f'sparsity {j+1}-derivative sigmoid')
		angles = sigmoid(diffs)
		plt.plot(taus[j+1:], angles, zorder=1)
		plt.scatter([taus[j+1:][index]], [angles[index]], color="r", zorder=2)


	plt.pause(100)


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

	total_sparsities = np.array(total_sparsities)
	total_sparsities /= (total_sparsities.max() - total_sparsities.min())
	total_sparsities *= 100
	if verbose:
		plt.figure(1)
		plt.title("tau vs. sparsity")
		plt.xlabel('tau')
		plt.ylabel('sparsity')
		vals = total_sparsities
		vals = sigmoid(total_sparsities)
		plt.plot(taus, vals)
		index = np.where(abs(taus-1)<1e-6)[0]
		plt.scatter([taus[index]], [vals[index]], color="r", zorder=2)

		plt.pause(10)
		
		if save:
			plt.savefig(tau_save_path, dpi=300, bbox_inches='tight')

		# plt.figure(2)
		# plt.title("alpha vs. sparsity")
		# plt.xlabel('alpha')
		# plt.ylabel('sparsity')
		# plt.plot(alphas, total_sparsities)
		# plt.pause(1)

		# if save:
		# 	plt.savefig(alpha_save_path, dpi=300, bbox_inches='tight')

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

