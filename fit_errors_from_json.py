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
from decision_tree_with_bagging import DecisionTreeWithBaggingRegressor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.pyplot import plot, ion, show
import random
from sklearn import metrics
import math
from dyadic_decision_tree_model import DyadicDecisionTreeModel
from sklearn.metrics import *
import json
from tqdm import tqdm

def get_pruned_m_step(cur_m_step, num_wavelets,  errors, error_TH=0.0):
	# print(f"errors:{len(errors)}, cur_m_step:{cur_m_step}")
	errors = np.array(errors)
	errors = errors[:cur_m_step]
	errors = errors[errors > error_TH]
	# print(f"errors:{len(errors)}, cur_m_step:{cur_m_step}")
	return len(errors)

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


def find_best_fit_alpha(errors_data, output_path, verbose=True):
	m_step, start_m_step = 10, 10
	errors = errors_data["errors"]
	n_wavelets = errors_data["n_wavelets"]	
	alphas, consts = [], []

	n_wavelets_log = np.log(np.reshape(n_wavelets, (-1, 1)))
	errors_log = np.log(np.reshape(errors, (-1, 1)))	
	
	normalize = True
	regr = linear_model.LinearRegression(normalize=normalize)
	# regr = linear_model.Ridge(alpha=0.5, normalize=normalize)
	# regr = linear_model.HuberRegressor(epsilon=1.5)

	ax = plt.axes()
	plt.title(f'log(#wavelets) to log(errors)')
	plt.xlabel('log(#wavelets)')
	plt.ylabel('log(errors)')
	plt.plot(n_wavelets_log, errors_log)
	plt.draw()
	stopping_percentile = 1.
	stop_condition = int(stopping_percentile*len(errors))

	print(f"number of errors:{len(errors)}")		
	for cur_m_step in tqdm(range(start_m_step, len(errors), m_step)):
		for start_m in range(0, 1, m_step):						
			pruned_m_step = get_pruned_m_step(cur_m_step, n_wavelets,  errors, error_TH=0.)
			pruned_m_step = min(stop_condition, pruned_m_step)

			cur_log_errors = errors_log[start_m:pruned_m_step]
			cur_log_wavelets = n_wavelets_log[start_m:pruned_m_step]

			regr.fit(cur_log_wavelets, cur_log_errors)
			cur_log_errors_pred = regr.predict(cur_log_wavelets)

			quality_of_fit = r2_score(cur_log_errors, cur_log_errors_pred)

			# import pdb; pdb.set_trace()
			try:
				alpha = np.abs(regr.coef_[0][0])		
				const = np.exp(regr.intercept_[0])
			except:
				alpha = np.abs(regr.coef_[0])		
				const = np.exp(regr.intercept_)

			
			alphas.append(alpha)
			consts.append(const)
			print(f"start:{start_m}, cur_m_step:{cur_m_step}, alpha:{alpha}, quality_of_fit:{quality_of_fit},  const:{const}")
			# continue

			if verbose:
				plt.figure(1)
				plt.plot(cur_log_wavelets, cur_log_errors_pred, color='blue', linewidth=3, label=f'alpha:{alpha}')
				plt.legend()
				plt.pause(1)
				ax.lines.pop(-1)

				plt.figure(2)
				plt.title("wavelet vs. errors")
				plt.xlabel('#wavelets')
				plt.ylabel('errors')
				plt.plot(n_wavelets[:cur_m_step], errors[:cur_m_step], color='g')
				plt.scatter(n_wavelets[pruned_m_step], errors[pruned_m_step], color='r')

	steps = list(range(start_m_step, len(errors), m_step))
	save = False
	if save:
		save_graphs(output_path, steps, alphas, consts)

if __name__ == "__main__":
	json_path = sys.argv[1]
	output_path = sys.argv[2]
	f = open(json_path)
	errors_data = json.load(f)
	find_best_fit_alpha(errors_data, output_path)

