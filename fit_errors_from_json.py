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

def get_pruned_m_step(cur_m_step, num_wavelets,  errors, error_TH=0.0):
	# print(f"errors:{len(errors)}, cur_m_step:{cur_m_step}")
	errors = np.array(errors)
	errors = errors[:cur_m_step]
	errors = errors[errors > error_TH]
	# print(f"errors:{len(errors)}, cur_m_step:{cur_m_step}")
	return len(errors)

def find_best_fit_alpha(errors_data):
	m_step, cur_m_step = 10, 10
	errors = errors_data["errors"]
	n_wavelets = errors_data["n_wavelets"]
	mean_norms = errors_data["mean_norms"]	

	n_wavelets_log = np.log(np.reshape(n_wavelets, (-1, 1)))
	errors_log = np.log(np.reshape(errors, (-1, 1)))
	mean_norms = np.reshape(mean_norms, (-1, 1))
	# regr = linear_model.LinearRegression()	
	regr = linear_model.Ridge(alpha=5.)
	ax = plt.axes()
	# ax  = plt.add_subplot(1, 1, 1)
	plt.title(f'log(#wavelets) to log(errors)')
	plt.xlabel('log(#wavelets)')
	plt.ylabel('log(errors)')
	plt.plot(n_wavelets_log, errors_log)
	plt.draw()

	print(f"number of errors:{len(errors)}")
	for cur_m_step in range(cur_m_step, len(errors), m_step):
		for start_m in range(0, 1, m_step):
			plt.figure(1)
			# start_m = 0 
			
			pruned_m_step = get_pruned_m_step(cur_m_step, n_wavelets,  errors, error_TH=0.)
			cur_log_errors = errors_log[start_m:pruned_m_step]
			cur_log_wavelets = n_wavelets_log[start_m:pruned_m_step]
			cur_mean_norms = mean_norms[start_m:pruned_m_step]

			# import pdb; pdb.set_trace()
			# regr.fit(cur_log_wavelets, cur_log_errors, cur_mean_norms.squeeze())
			regr.fit(cur_log_wavelets, cur_log_errors)
			cur_log_errors_pred = regr.predict(cur_log_wavelets)

			quality_of_fit = r2_score(cur_log_errors, cur_log_errors_pred)

			alpha = np.abs(regr.coef_[0][0])		
			const = np.exp(regr.intercept_[0])
			print(f"start:{start_m}, cur_m_step:{cur_m_step}, alpha:{alpha}, quality_of_fit:{quality_of_fit},  const:{const}")
			# continue

			plt.plot(cur_log_wavelets, cur_log_errors_pred, color='blue', linewidth=3, label=f'alpha:{alpha}')
			plt.legend()
			plt.pause(1)
			ax.lines.pop(-1)

			plt.figure(2)
			plt.plot(n_wavelets[:cur_m_step], errors[:cur_m_step], color='blue')

if __name__ == "__main__":
	json_path = sys.argv[1]
	f = open(json_path)
	errors_data = json.load(f)
	find_best_fit_alpha(errors_data)

