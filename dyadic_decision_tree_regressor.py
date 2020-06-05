import os 
import sys
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
# ion() # enables interactive mode

# f1 = plt.figure(1)
# f2 = plt.figure(2)

class DyadicDecisionTreeRegressor:
	def __init__(self, depth=9, seed=None, norms_normalization='volume', cube_length=1.):
		'''
		Construct a new 'WaveletsForestRegressor' object.

		:regressor: Regressor type. Either "rf" or "decision_tree_with_bagging". Default is "rf".
		:criterion: Splitting criterion. Same options as sklearn\'s DecisionTreeRegressor. Default is "mse".
		:bagging: Bagging. Only available when using the "decision_tree_with_bagging" regressor. Default is 0.8.
		:depth: Maximum depth of each tree. Default is 9.
		:trees: Number of trees in the forest. Default is 5.
		:features: Features to consider in each split. Same options as sklearn\'s DecisionTreeRegressor.
		:seed: Seed for random operations. Default is 2000.
		'''

		self.norms = None
		self.vals = None		
		##
		self.volumes = None
		self.X = None
		self.y = None
		self.rf = None

		self.cube_length = cube_length
		self.depth = depth

		self.regressor = DyadicDecisionTreeModel(depth=self.depth, cube_length=self.cube_length)
		
		self.seed = seed		
		self.norms_normalization = norms_normalization	

	# def __compute_norm(self, avg, parent_avg, volume):      
	# 	norm = np.sqrt(np.sum(np.square(avg - parent_avg)) * volume)
	# 	return norm

	def print_regressor(self):
		print(self.regressor.print_tree())

	def compute_average_score_from_tree(self, tree_value):
		if self.mode == 'classification':
			y_vec = [-1. , 1.]
			result = tree_value.dot(y_vec)/tree_value.sum()         
			return result
		else:
			return tree_value[:, 0]

	def __traverse_nodes(self, estimator, base_node_id, node_box, norms, vals, rectangles, levels):
		# https://stackoverflow.com/questions/47719001/what-does-scikit-learn-decisiontreeclassifier-tree-value-do
		# https://stackoverflow.com/questions/52376272/getting-the-value-of-a-leaf-node-in-a-decisiontreeregressor
		
		# feature - feature used for splitting!
		# INTERSECTION: [LEFT, RIGHT, DOWN, UP]		

		if base_node_id == 0:
			vals[:, base_node_id] = self.compute_average_score_from_tree(estimator.tree_.value[base_node_id])
			norms[base_node_id] = self.__compute_norm(vals[:, base_node_id], 0, 1)			
			rectangles[base_node_id] = np.array([0., 1., 0., 1.])

		left_id = estimator.tree_.children_left[base_node_id]
		right_id = estimator.tree_.children_right[base_node_id]		

		if left_id >= 0:			
			rectangles[left_id] = rectangles[base_node_id]
			levels[left_id] = levels[base_node_id] + 1
			tree = estimator.tree_			
			left_feature = tree.feature[base_node_id]
			left_threshold = tree.threshold[base_node_id]
			
			if left_feature == 0:
				rectangles[left_id][1] = left_threshold
			else:				
				rectangles[left_id][3] = left_threshold			

			node_box[left_id, :, :] = node_box[base_node_id, :, :]			
			# import pdb; pdb.set_trace()
			node_box[left_id, estimator.tree_.feature[base_node_id], 1] = np.min(
				[estimator.tree_.threshold[base_node_id], node_box[left_id, estimator.tree_.feature[base_node_id], 1]])
			self.__traverse_nodes(estimator, left_id, node_box, norms, vals, rectangles, levels)
			vals[:, left_id] = self.compute_average_score_from_tree(estimator.tree_.value[left_id]) - \
				self.compute_average_score_from_tree(estimator.tree_.value[base_node_id])
			norms[left_id] = self.__compute_norm(vals[:, left_id], vals[:, base_node_id], 1)

		if right_id >= 0:
			rectangles[right_id] = rectangles[base_node_id]
			levels[right_id] = levels[base_node_id] + 1
			tree = estimator.tree_
			right_feature = tree.feature[base_node_id]
			right_threshold = tree.threshold[base_node_id]
			
			if left_feature == 0:
				rectangles[right_id][0] = right_threshold
			else:
				rectangles[right_id][2] = right_threshold

			node_box[right_id, :, :] = node_box[base_node_id, :, :]
			node_box[right_id, estimator.tree_.feature[base_node_id], 0] = np.max(
				[estimator.tree_.threshold[base_node_id], node_box[right_id, estimator.tree_.feature[base_node_id], 0]])
			self.__traverse_nodes(estimator, right_id, node_box, norms, vals, rectangles, levels)
			vals[:, right_id] = self.compute_average_score_from_tree(estimator.tree_.value[right_id]) - \
				self.compute_average_score_from_tree(estimator.tree_.value[base_node_id])
			norms[right_id] = self.__compute_norm(vals[:, right_id], vals[:, base_node_id], 1)

	def fit(self, X_raw, y):
		self.regressor.fit(X_raw)

	##

	# def predict(self, X, m=1000, start_m=0, paths=None):
	# 	'''
	# 	Predict using a maximum of M-terms
	# 	:X: Data samples.
	# 	:m: Maximum of M-terms.
	# 	:start_m: The index of the starting term. Can be used to evaluate predictions incrementally over terms.paths.shape
	# 	:paths: Instead of computing decision paths for each sample, the method can receive the indicator matrix. Can be used to evaluate predictions incrementally over terms.
	# 	:return: Predictions.
	# 	'''

	# 	sorted_norms = np.argsort(-self.norms)[start_m:m]
	# 	if paths == None:
	# 		paths, n_nodes_ptr = self.rf.decision_path(X)
	# 	pruned = lil_matrix(paths.shape, dtype=np.float32)
	# 	pruned[:, sorted_norms] = paths[:, sorted_norms]
	# 	predictions = pruned * self.vals.T / len(self.rf.estimators_)
	# 	return predictions

	# def evaluate_smoothness(self, m=1000):
	# 	'''
	# 	Evaluates smoothness for a maximum of M-terms
	# 	:m: Maximum terms to use. Default is 1000.
	# 	:return: Smothness index, n_wavelets, errors.
	# 	'''
	# 	n_wavelets = []
	# 	errors = []
	# 	step = 10
	# 	power = 2
	# 	print_errors = False

	# 	paths, n_nodes_ptr = self.rf.decision_path(self.X)
	# 	predictions = np.zeros(np.shape(self.y))
	# 	for m_step in range(2, m, step):
	# 		if m_step > len(self.norms):
	# 			break
	# 		predictions += self.predict(self.X, m=m_step, start_m=max(m_step - step, 0), paths=paths)                        
	# 		error_norms = np.power(np.sum(np.power(self.y - predictions, power), 1), 1. / power)
	# 		total_error = np.sum(np.square(error_norms), 0) / len(self.X)
			
	# 		if m_step > 0 and total_error > 0:
	# 			if print_errors:
	# 				logging.info('Error for m=%s: %s' % (m_step - 1, total_error))
	# 			n_wavelets.append(m_step - 1)
	# 			errors.append(total_error)
	# 	logging.info(f"total m_step is {m_step}")        


	# 	plt.figure(1)        
	# 	plt.clf()
	# 	n_wavelets = np.reshape(n_wavelets, (-1, 1))
	# 	errors = np.reshape(errors, (-1, 1))
	# 	plt.title(f'#wavelets to errors')
	# 	plt.xlabel('#wavelets')
	# 	plt.ylabel('errors')
	# 	plt.plot(n_wavelets, errors)
	# 	plt.draw()
	# 	plt.pause(1)		

	# 	plt.figure(2)
	# 	plt.clf()
	# 	n_wavelets_log = np.log(np.reshape(n_wavelets, (-1, 1)))
	# 	errors_log = np.log(np.reshape(errors, (-1, 1)))
	# 	plt.title(f'log(#wavelets) to log(errors)')
	# 	plt.xlabel('log(#wavelets)')
	# 	plt.ylabel('log(errors)')
	# 	plt.plot(n_wavelets_log, errors_log)
		
	# 	regr = linear_model.LinearRegression()
	# 	# regr = linear_model.Ridge(alpha=.8)
		
	# 	regr.fit(n_wavelets_log, errors_log)

	# 	y_pred = regr.predict(n_wavelets_log)
	# 	plt.plot(n_wavelets_log, y_pred, color='blue', linewidth=3)
	# 	plt.draw()
	# 	plt.pause(1)

	# 	alpha = np.abs(regr.coef_[0][0])
	# 	# logging.info('Smoothness index: %s' % alpha)

	# 	return alpha, n_wavelets, errors

	def accuracy(self, y_pred, y):
		'''
		Evaluates accuracy given predictions and actual labels.

		:y_pred: Predictions as vertices on the simplex (preprocessed by 'pred_to_one_hot').
		:y: Actual labels.
		:return: Accuracy.
		'''
		return accuracy_score(y, y_pred)

	def pred_to_one_hot(self, y_pred):
		'''
		Converts regression predictions to their closest vertices on the simplex
		'''
		argmax = np.argmax(y_pred, 1)
		ret = np.zeros((len(argmax), np.shape(y_pred)[1]))
		ret[np.arange(len(argmax)), argmax] = 1
		return ret


if __name__ == '__main__':
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.datasets import make_classification	
	from sklearn import tree
	X = [[0, 0], [1, 1]]
	Y = [0, 1]  
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(X, Y)
