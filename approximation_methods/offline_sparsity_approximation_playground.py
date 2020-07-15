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
	N = 1
	total_sparsities = np.array(total_sparsities)
	# total_sparsities /= 10000
	save = False
	tau_save_path = os.path.join(output_path, "tau_sparsity_derivatives.png")
	alpha_save_path = os.path.join(output_path, "alpha_sparsity_derivatives.png")
	h = 0.01
	# first order
	diffs = np.array(total_sparsities[1:]) - np.array(total_sparsities[:-1])	

	for j in range(N):			
		if j != 0:
			diffs = np.array(diffs[1:]) - np.array(diffs[:-1])			
		# if j < N-2:
		# 	continue

		diffs /= h

		# plt.figure(j+4)
		# plt.title(f"tau vs. sparsity {j+1}-derivative")
		# plt.xlabel(f'tau')
		# plt.ylabel(f'sparsity {j+1}-derivative')		

		index = np.where(abs(taus[j+1:]-1)<1e-6)[0]
		# indices = np.where(np.logical_and(abs(taus[j+1:]-1)<0.1, taus[j+1:]>1))
		# # plt.ylim(-150000, 0)
		# plt.plot(taus[j+1:], diffs, zorder=1)
		# plt.scatter([taus[j+1:][indices]], [diffs[indices]], color="r", zorder=2, \
		# 	s=[2 for k in range(len(indices))])
		# plt.scatter([taus[j+1:][index]], [diffs[index]], color="b", zorder=3)

		plt.figure(j+1)
		plt.title(f"tau vs. sparsity {j+1}-derivative angle")
		plt.xlabel(f'tau')
		plt.ylabel(f'sparsity {j+1}-derivative angle')
		angles = np.rad2deg(np.arctan(diffs))		

		plt.plot(taus[j+1:], angles, zorder=1)
		plt.scatter([taus[j+1:][index]], [angles[index]], color="r", zorder=2)

		# import pdb; pdb.set_trace()
		angle_index = np.where(abs(angles+(90. - 0.015))<1e-2)[0][0]
		print(f"[taus[j+1:][angle_index]]:{[taus[j+1:][angle_index]]}")
		plt.scatter([taus[j+1:][angle_index]], [angles[angle_index]], color="g", zorder=2)
		

		# plt.title(f"tau vs. sparsity {j+1}-derivative sigmoid")
		# plt.xlabel(f'tau')
		# plt.ylabel(f'sparsity {j+1}-derivative sigmoid')
		# angles = sigmoid(diffs)
		# plt.plot(taus[j+1:], angles, zorder=1)
		# plt.scatter([taus[j+1:][index]], [angles[index]], color="r", zorder=2)


	plt.pause(2000)


def find_best_fit_alpha(errors_data, output_path, verbose=True, p=2):
	m_step, start_m_step = 10, 10
	norms = np.array(errors_data["norms"])
	tau_save_path = os.path.join(output_path, "tau_sparsity.png")
	alpha_save_path = os.path.join(output_path, "alpha_sparsity.png")

	taus = np.arange(0.7, 20., 0.01)
	alphas = ((1/taus) - 1/p)
	save = True	
	total_sparsities, total_alphas = [], []
	
	for tau in taus:
		tau_sparsity = np.power(np.power(norms, tau).sum(), (1/tau))
		total_sparsities.append(tau_sparsity)
		print(f"tau:{tau}, tau_sparsity:{tau_sparsity}")

	total_sparsities = np.array(total_sparsities)
	# total_sparsities /= (total_sparsities.max() - total_sparsities.min())
	# total_sparsities *= 100
	if verbose:
		plt.figure(1)
		plt.title("tau vs. sparsity")
		plt.xlabel('tau')
		plt.ylabel('sparsity')
		# plt.yscale('log')		
		
		window = 2500
		vals = total_sparsities


		# vals = [vals[idx : min(len(vals)-1, idx + window//2)].mean() \
		# 	if idx != len(vals)-1 else vals[idx] for idx, k in enumerate(vals)]

		plt.plot(taus, vals)
		index = np.where(abs(taus-1)<1e-6)[0][0]							
		plt.scatter([taus[index]], vals[index], color="b", zorder=3)

		# x = np.linspace(50, list(range(50,351,25))[-1], 25)
		TH = 1000
		y = [TH for k in taus]
		plt.plot(taus, y, '-r', label=f'TH:{TH}')
		plt.pause(1000)
		
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

	plot_derivatives(taus, alphas, total_sparsities, output_path)

		

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

