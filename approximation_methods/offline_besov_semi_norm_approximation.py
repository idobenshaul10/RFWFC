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


def find_best_fit_alpha(errors_data, output_path, verbose=True, p=2):
	m_step, start_m_step = 10, 10
	summands = np.array(errors_data["summands"])

	save_path = os.path.join(output_path, "besov_semi_norm.png")
	alphas = np.arange(0.05, 1., 0.01)
	total_besov_norms = []
	for alpha in alphas:
		
		tau = 1/(alpha + (1/p))
		tau_summands = np.power(summands, tau)
		besov_norm_approx = np.power(tau_summands.sum(), 1/tau)
		total_besov_norms.append(besov_norm_approx)

	if verbose:
		plt.figure(1)
		plt.title("alpha vs. semi_norm")
		plt.xlabel('alpha')
		plt.ylabel('semi_norm')
		plt.plot(alphas, total_besov_norms)		
		plt.pause(10)
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		# ax.lines.pop(-1)		

if __name__ == "__main__":
	json_path = sys.argv[1]
	output_path = None
	if len(sys.argv) > 2:
		output_path = sys.argv[2]
	else:		
		output_path = os.path.dirname(sys.argv[1])
	f = open(json_path)
	errors_data = json.load(f)
	find_best_fit_alpha(errors_data, output_path)

