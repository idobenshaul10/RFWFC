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
from sklearn.metrics import r2_score
import json


def find_best_fit_alpha(errors_data):
	m_step, cur_m_step = 10, 10
	errors = errors_data["errors"]
	n_wavelets = errors["n_wavelets"]
	n_wavelets_log = np.log(np.reshape(n_wavelets, (-1, 1)))
	errors_log = np.log(np.reshape(errors, (-1, 1)))
	regr = linear_model.LinearRegression()

	for cur_m_step in range(0, len(errors), m_step):
		cur_log_errors = errors_log[:cur_m_step]
		cur_log_wavelets = n_wavelets_log[:cur_m_step]				
		regr.fit(cur_log_wavelets, cur_log_errors)


		cur_log_errors_pred = regr.predict(cur_log_wavelets)
		quality_of_fit = r2_score(cur_log_errors, cur_log_errors_pred)

		alpha = np.abs(regr.coef_[0][0])
		const = regr.coef_[0][0]
		print(f"cur_m_step:{cur_m_step}, alpha:{alpha}, quality_of_fit:{quality_of_fit},  const:{const}")


if __name__ == "__main__":
	json_path = sys.argv[1]
	errors_data = json.load(json_path)
	find_best_fit_alpha(errors_data)

