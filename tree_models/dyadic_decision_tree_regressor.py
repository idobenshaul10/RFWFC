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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.pyplot import plot, ion, show
import random
from sklearn import metrics
import math
from tree_models.decision_tree_with_bagging import DecisionTreeWithBaggingRegressor
from tree_models.dyadic_decision_tree_model import DyadicDecisionTreeModel
import json
from tqdm import tqdm

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
		self.verbose = False
		self.save_errors = False
		self.save_semi_norm = True
		self.power = 2

		self.cube_length = cube_length
		self.depth = depth

		self.regressor = DyadicDecisionTreeModel(depth=self.depth, \
			cube_length=self.cube_length, verbose=self.verbose)
		
		self.seed = seed		
		self.norms_normalization = norms_normalization	

	def print_regressor(self):
		self.regressor.test_tree_indices()
		print(self.regressor.print_tree())

	def fit(self, X_raw, y):
		# if self.verbose:
		# 	self.print_regressor()
		self.regressor.add_dataset(X_raw, y)
		self.regressor.fit(X_raw)
		self.regressor.init_ids()

		self.X = X_raw
		self.y = y		
		
		try:
			val_size = np.shape(y)[1]
		except:
			val_size = 1		

		num_nodes = len(self.regressor.nodes)
		num_features = np.shape(X_raw)[1]

		norms = np.zeros(num_nodes)
		vals = np.zeros((val_size, num_nodes))
		self.__traverse_nodes(0, norms, vals) # 50		
		
		num_samples = np.array([node.num_samples for node in self.regressor.nodes])		
		norms = np.multiply(norms, np.power(num_samples, 1/self.power))

		self.epsilon = 1e-6
		# we need to remove the nodes with <epsilon norm!
		self.non_zero_norm_indices = (norms >= self.epsilon).astype(np.int32)
		self.regressor.non_zero_norm_indices = self.non_zero_norm_indices

		self.norms = norms
		self.vals = vals
	
	def __compute_norm(self, avg, parent_avg, volume):
		# norm = np.sqrt(np.sum(np.square(avg - parent_avg)) * volume)
		# print(avg - parent_avg, norm)		
		norm = np.power(np.sum(np.power(np.abs(avg - parent_avg), self.power)) * volume, (1/self.power))
		return norm

	def __traverse_nodes(self, base_node_id, norms, vals):		
		parent_node = self.regressor.nodes[base_node_id]
		parent_mean_value = self.regressor.get_mean_value(base_node_id)

		if base_node_id == 0:
			vals[:, base_node_id] = parent_mean_value
			norms[base_node_id] = self.__compute_norm(vals[:, base_node_id], 0, volume=1)		
		
		if hasattr(parent_node, 'left'):		
			left_id = parent_node.left.id
			if left_id >= 0:
				self.__traverse_nodes(left_id, norms, vals)	
				mean_left_value = self.regressor.get_mean_value(left_id)				
				vals[:, left_id] = mean_left_value - parent_mean_value
				norms[left_id] = self.__compute_norm(vals[:, left_id], vals[:, base_node_id], volume=1)		

		if hasattr(parent_node, 'right'):
			right_id = parent_node.right.id
			if right_id >= 0:
				self.__traverse_nodes(right_id, norms, vals)
				mean_right_value = self.regressor.get_mean_value(right_id)				
				vals[:, right_id] = mean_right_value - parent_mean_value
				norms[right_id] = self.__compute_norm(vals[:, right_id], vals[:, base_node_id], volume=1)		
	

	def predict(self, X, m=1000, start_m=0, paths=None):
		'''
		Predict using a maximum of M-terms
		:X: Data samples.
		:m: Maximum of M-terms.
		:start_m: The index of the starting term. Can be used to evaluate predictions incrementally over terms.paths.shape
		:paths: Instead of computing decision paths for each sample, the method can receive the indicator matrix. Can be used to evaluate predictions incrementally over terms.
		:return: Predictions.
		'''		
		sorted_norms = np.argsort(-self.norms)[start_m:m]
		if paths is None:
			paths = self.regressor.decision_path(X)
		
		pruned = lil_matrix(paths.shape, dtype=np.float32)
		pruned[:, sorted_norms] = paths[:, sorted_norms]		

		predictions = pruned * self.vals.T
		return predictions, sorted_norms

	def evaluate_angle_smoothness(self, m=1000, error_TH=0):
		'''
		Evaluate smoothness using sparsity consideration
		'''
		approx_diff = False
		norms = list(self.norms[self.non_zero_norm_indices==1])		
		norms = norms[1:]
		p = 2
		h = 0.01
		diffs = []		
		taus = np.arange(0.7, 10., h)		
		total_sparsities, total_alphas = [], []
		if approx_diff:
			for tau in taus:
				tau_sparsity = np.power(np.power(norms, tau).sum(), (1/tau))
				total_sparsities.append(tau_sparsity)				
			total_sparsities = np.array(total_sparsities)
			diffs = total_sparsities[1:] - total_sparsities[:-1]
			diffs /= h
		else:
			for tau in taus:		
				tau_sparsity = np.power(np.power(norms, tau).sum(), ((1/tau)-1))
				tau_sparsity *= np.power(norms, (tau-1)).sum()
				diffs.append(tau_sparsity)				
			diffs = -np.array(diffs)

		angles = np.rad2deg(np.arctan(diffs))		
		epsilon_1 = 0.075
		epsilon_2 = 2*epsilon_1
		
		epsilon_1_indices = np.where(abs(angles+90.)<=epsilon_1)[0]
		epsilon_2_indices = np.where(abs(angles+90.)<=epsilon_2)[0]			

		angle_index_1 = epsilon_1_indices[int(0.75*len(epsilon_1_indices))]
		angle_index_2 = epsilon_2_indices[int(0.75*len(epsilon_2_indices))]

		critical_tau_approximation_1 = taus[1:][angle_index_1]
		critical_alpha_approximation_1 = ((1/critical_tau_approximation_1) - 1/p)

		critical_tau_approximation_2 = taus[1:][angle_index_2]
		critical_alpha_approximation_2 = ((1/critical_tau_approximation_2) - 1/p)

		return critical_alpha_approximation_1, critical_alpha_approximation_2


	def evaluate_smoothness(self, m=1000, error_TH=0):
		'''
		Evaluates smoothness for a maximum of M-terms
		:m: Maximum terms to use. Default is 1000.
		:return: Smothness index, n_wavelets, errors.
		'''
		n_wavelets = []
		errors = []
		mean_norms = []
		step = 10		
		print_errors = False

		paths = self.regressor.decision_path(self.X)		
		predictions = np.zeros(self.y.shape)		

		# , total=self.non_zero_norm_indices.sum()
		num_non_zero_nodes = self.non_zero_norm_indices.sum()
		for m_step in tqdm(range(2, m, step), total=num_non_zero_nodes):
			if m_step > num_non_zero_nodes:
				break
			
			start_m = max(m_step - step, 0)
			pred_result, sorted_norms = self.predict(self.X, m=m_step, start_m=start_m, paths=paths)			
			predictions += pred_result
			error_norms = np.power(np.sum(np.power(np.abs(self.y - predictions), self.power), 1), (1./self.power))			
			total_error = np.sum(np.power(np.abs(error_norms), 2), 0) / len(self.X)
			# total_error = np.sum(np.power(np.abs(error_norms), 1), 0) / len(self.X)
			
			if len(errors)> 0:
				if errors[-1] == total_error:
					break

			if total_error < error_TH:
				break
			
			if m_step > 0 and total_error > 0:
				if print_errors:
					print(f"m_step:{m_step}, total_error:{total_error}")					
				n_wavelets.append(m_step - 1)
				errors.append(total_error)
				mean_norms.append(sorted_norms.mean())
		
		if self.verbose:
			plt.figure(1)
			plt.clf()
		n_wavelets = np.reshape(n_wavelets, (-1, 1))
		errors = np.reshape(errors, (-1, 1))		

		if self.verbose:			
			plt.title(f'#wavelets to errors, Num Wavelets:{m}')
			plt.xlabel('#wavelets')
			plt.ylabel('errors')		
			plt.plot(n_wavelets, errors)
			plt.draw()
			plt.pause(0.5)
			plt.figure(2)
			plt.clf()
		n_wavelets_log = np.log(np.reshape(n_wavelets, (-1, 1)))
		errors_log = np.log(np.reshape(errors, (-1, 1)))
		if self.verbose:
			# plt.title(f'#wavelets to errors, Num Wavelets:{m}')
			plt.title(f'log(#wavelets) to log(errors), Num Wavelets:{m}')
			plt.xlabel('log(#wavelets)')
			plt.ylabel('log(errors)')
			plt.plot(n_wavelets_log, errors_log)

		if self.save_errors:
			def convert(o):
				if isinstance(o, np.generic): return o.item()  
				raise TypeError

			dir_path = r"C:\projects\RFWFC\results\Jackson_Round_2"
			json_file_name = "50000_points_new.json"
			write_data = {}			
			write_data['n_wavelets'] = list(n_wavelets.squeeze())
			write_data['errors'] = list(errors.squeeze())			
			write_data['mean_norms'] = list(mean_norms)
			with open(os.path.join(dir_path, json_file_name), "w+") as f:
				json.dump(write_data, f, default=convert)
		
		regr = linear_model.LinearRegression()
		regr.fit(n_wavelets_log, errors_log)

		y_pred = regr.predict(n_wavelets_log)
		alpha = np.abs(regr.coef_[0][0])

		if self.verbose:
			plt.plot(n_wavelets_log, y_pred, color='blue', linewidth=3, label=f'alpha:{alpha}')
			plt.legend()
			plt.draw()
			plt.pause(0.5)
			logging.info('Smoothness index: %s' % alpha)

		return alpha, n_wavelets, errors

	def save_wavelet_norms(self):
		result = list(self.norms[self.non_zero_norm_indices==1])
		# remove root node
		result = result[1:]
		
		def convert(o):
			if isinstance(o, np.generic): return o.item()
			raise TypeError

		dir_path = r"C:\projects\RFWFC\results\approximation_methods\Sparsity"
		json_file_name = "norms_50000.json"
		write_data = {}
		write_data['norms'] = result
		norms_path = os.path.join(dir_path, json_file_name)
		with open(norms_path, "w+") as f:
			json.dump(write_data, f, default=convert)

		print(f"saved norms to:{norms_path}")		

	def calculate_besov_semi_norm(self, p=2, volume_method="num_samples"):
		total_domain_scores = []
		for domain in self.regressor.nodes:
			idx = domain.id
			if self.non_zero_norm_indices[idx] == 0:
				continue
				
			if volume_method == "num_samples":				
				volume = np.power(domain.num_samples, 1/p)
			else:
				volume = np.power(domain.rect.volume, 1/p)

			num_samples = domain.num_samples
			mean_value = self.regressor.get_mean_value(idx)
			ys = self.regressor.y_all[domain.indices]
			diff_from_mean = np.abs(ys - mean_value).sum()
			diff_from_mean /= num_samples
			domain_score = volume*diff_from_mean
			total_domain_scores.append(domain_score)

		def convert(o):
			if isinstance(o, np.generic): return o.item()  
			raise TypeError

		dir_path = r"C:\projects\RFWFC\results\approximation_methods\Besov_Semi_Norm"
		json_file_name = f"besov_summands_50000_{volume_method}.json"
		write_data = {}		
		write_data['summands'] = total_domain_scores
		summands_path = os.path.join(dir_path, json_file_name)
		with open(summands_path, "w+") as f:
			json.dump(write_data, f, default=convert)
		print(f"saved summands to:{summands_path}")
		
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
