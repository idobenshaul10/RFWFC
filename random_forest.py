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
from matplotlib.pyplot import plot, ion, show
import random
from sklearn import metrics
import math
import json
from tqdm import tqdm
# ion() # enables interactive mode

# f1 = plt.figure(1)
# f2 = plt.figure(2)

class WaveletsForestRegressor:
	def __init__(self, regressor='random_forest', mode='classification', criterion='gini', bagging=0.8, train_vi=False,
				 depth=9, trees=5, features='auto', seed=None, vi_threshold=0.8, \
				 norms_normalization='volume', cube_length=1.):
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
		self.power = 2
		##
		self.si_tree = None
		self.si = None
		self.feature_importances_ = None
		##
		self.volumes = None
		self.X = None
		self.y = None
		self.rf = None

		self.mode = mode
		self.regressor = regressor
		self.criterion = criterion
		self.bagging = bagging

		if self.regressor == "random_forest" and depth == -1:
			self.depth = None
		else:
			self.depth = depth
		self.trees = trees
		self.seed = seed
		self.train_vi = train_vi
		self.vi_threshold = vi_threshold
		self.norms_normalization = norms_normalization
		self.save_errors = False


	def visualize_classifier(self, ax=None, cmap='rainbow', depth=-1):        
		# ion()     
		ax = ax or plt.gca()
		colors = self.y.reshape(-1)     
		
		indices = np.random.choice(self.X.shape[0], int(len(self.X)/5))
		show_X = self.X[indices]

		# ax.scatter(show_X[:, 0], show_X[:, 1], c=colors[indices], \
		#   clim=(self.y.min(), self.y.max()), s=0.1, cmap=cmap, zorder=1)

		ax.axis('tight')        
		xlim = ax.get_xlim()
		ylim = ax.get_ylim()        
		xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
							 np.linspace(*ylim, num=200))

		circle2 = plt.Circle((0.5, 0.5), 0.4, color='b', fill=False, lw=0.25)       
		ax.add_artist(circle2)
		
		Z = self.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)       

		n_classes = len(np.unique(Z))
		contours = ax.contourf(xx, yy, Z, alpha=0.3,
							   levels=np.arange(n_classes + 1) - 0.5,
							   cmap=cmap, zorder=1)
		ax.set(xlim=xlim, ylim=ylim)        
		plt.show(block=True)

		# dir_path = r"C:\projects\RFWFC\results\decision_tree_with_bagging\visualizations\per_depth"
		# save_path = os.path.join(dir_path, f"circle_{depth}.png")
		# print(f"save_path:{save_path}")
		# plt.savefig(save_path, \
		#     dpi=300, bbox_inches='tight')
		# plt.clf()

	def get_volumes(self, rectangles):
		result = []
		for rectangle in rectangles:
			l, r, d, u = rectangle
			result.append(abs(l-r)*abs(d-u))
		return np.array(result)

	def calculate_level_volumes(self, rectangles, levels):
		levels_volumes = np.zeros(len(np.unique(levels)))
		intersections = self.find_rectangle_intersection(rectangles)
		volumes = self.get_volumes(rectangles)
		intersection_volumes = intersections*volumes
		for i in range(len(intersection_volumes)):
			levels_volumes[int(levels[i])] += intersection_volumes[i]
		
		levels_volumes
		return levels_volumes

	
	def collision(self, rleft, rtop, width, height, center_x, center_y, radius):  
		
		rright, rbottom = rleft + width/2, rtop + height/2		
		cleft, ctop     = center_x-radius, center_y-radius
		cright, cbottom = center_x+radius, center_y+radius
		
		if rright < cleft or rleft > cright or rbottom < ctop or rtop > cbottom:
			return False  

		
		for x in (rleft, rleft+width):
			for y in (rtop, rtop+height):				
				if math.hypot(x-center_x, y-center_y) <= radius:
					return True  # collision detected

		
		if rleft <= center_x <= rright and rtop <= center_y <= rbottom:
			return True

		return False


	def find_rectangle_intersection(self, rectangles):
		# INTERSECTION: [LEFT, RIGHT, DOWN, UP]
		intersections = np.zeros(rectangles.shape[0])
		for idx, rectangle in enumerate(rectangles):
			l, r, d, u = rectangle
			does_intersect = int(self.collision(l, u, r-l, u-d, 0, 0, 1))
			intersections[idx] = does_intersect

		# intersections = np.zeros(rectangles.shape[0])
		# for idx, rectangle in enumerate(rectangles):
		# 	l, r, d, u = rectangle
		# 	top_right = r*r + u*u > 1
		# 	top_left = l*l + u*u > 1
		# 	bottom_right = r*r + d*d > 1
		# 	bottom_left = l*l + d*d > 1
		# 	intersections[idx] = int(not(top_right and top_left and bottom_right and bottom_left))
		return intersections

	def fit(self, X_raw, y):
		'''
		Fit non-normalized data to simplex labels.

		:X_raw: Non-normalized features, given as a 2D array with each row representing a sample.
		:y: Labels, each row is given as a vertex on the simplex.
		'''

		logging.info('Fitting %s samples' % np.shape(X_raw)[0])
		X = (X_raw - np.min(X_raw, 0))/(np.max(X_raw, 0) - np.min(X_raw, 0))
		X = np.nan_to_num(X)
		# X = X_raw
		
		self.X = X
		self.y = y

		regressor = None
		if self.regressor == 'decision_tree_with_bagging':
			regressor = DecisionTreeWithBaggingRegressor(
				bagging=self.bagging,
				criterion=self.criterion,
				depth=self.depth,
				trees=self.trees,
				seed=self.seed,
			)
		else:
			# RandomForestRegressor
			# RandomForestClassifier
			if self.mode == 'classification':
				regressor = ensemble.RandomForestClassifier(
					criterion='gini',
					n_estimators=self.trees, 
					max_depth=self.depth,
					max_features='auto',
					n_jobs=-1,
					random_state=self.seed,
				)

			elif self.mode == 'regression':
				regressor = ensemble.RandomForestRegressor(
					n_estimators=self.trees, 
					max_depth=self.depth,
					max_features='auto',
					n_jobs=-1,
					random_state=self.seed,
				)
			else:
				print("ERROR, WRONG MODE")
				exit()

		rf = regressor.fit(self.X, self.y)
		self.rf = rf

		# y_pred = self.rf.predict(X)         
		# auc = metrics.roc_auc_score(y, y_pred)
		# print(f"auc:{auc}")     

		try:
			val_size = np.shape(y)[1]
		except:
			val_size = 1
		self.norms = np.array([])
		self.vals = np.zeros((val_size, 0))
		self.volumes = np.array([])
		##
		self.num_samples = np.array([])
		self.si_tree = np.zeros((np.shape(X)[1], len(rf.estimators_)))
		self.si = np.array([])
		self.root_nodes = []
		##

		for i in range(len(rf.estimators_)):
			# logging.info('Working on tree %s' % i)
			estimator = rf.estimators_[i]
			num_nodes = len(estimator.tree_.value)
			num_features = np.shape(X)[1]
			node_box = np.zeros((num_nodes, num_features, 2))
			node_box[:, :, 1] = 1

			norms = np.zeros(num_nodes)
			vals = np.zeros((val_size, num_nodes))          
			
			# INTERSECTION: [LEFT, RIGHT, DOWN, UP]
			rectangles = np.zeros((norms.shape[0], 4))			
			levels = np.zeros((norms.shape[0]))
			self.__traverse_nodes(estimator, 0, node_box, norms, vals, rectangles, levels)
			self.calculate_level_volumes(rectangles, levels)


			# logging.info('Traversing nodes of tree %s to extract volumes and norms' % i)			
			volumes = np.product(node_box[:, :, 1] - node_box[:, :, 0], 1)

			paths = estimator.decision_path(X)
			paths_fullmat = paths.todense()
			num_samples = np.sum(paths_fullmat, 0)/paths_fullmat.shape[0]

			if self.norms_normalization == 'volume':
				norms = np.multiply(norms, np.power(volumes, 1/self.power))
			else:
				norms = np.multiply(norms, np.power(num_samples, 1/self.power))
			# logging.info('Number of wavelets in tree %s: %s' % (i, np.shape(norms)[0]))


			self.volumes = np.append(self.volumes, volumes)
			self.norms = np.append(self.norms, norms)
			if len(self.root_nodes) == 0:
				self.root_nodes.append(0)
			else:
				self.root_nodes.append(self.root_nodes[-1] + num_nodes)

			self.num_samples = np.append(self.num_samples, num_samples)
			self.vals = np.append(self.vals, vals, axis=1)          
			##
			if self.train_vi:
				for k in range(0, num_features):
					vi_node_box = np.zeros((num_nodes, num_features, 2))
					vi_node_box[:, :, 1] = 1
					vi_norms = np.zeros(num_nodes)
					vi_vals = np.zeros((val_size, num_nodes))
					self.__variable_importance(estimator, 0, vi_node_box, vi_norms, vi_vals, k, self.vi_threshold)
					if self.norms_normalization == 'volume':
						vi_norms = np.multiply(vi_norms, np.sqrt(volumes))
					else:
						vi_norms = np.multiply(vi_norms, np.sqrt(num_samples))
					self.si_tree[k, i] = np.sum(vi_norms)

		self.si = np.append(self.si, np.sum(self.si_tree, 1) / len(rf.estimators_))
		self.feature_importances_ = self.si
		##
		y_pred_2 = self.predict(X)
		# auc = metrics.roc_auc_score(y, y_pred_2)
		# print(f"AUC_2:{auc}")       
		return self

	def __compute_norm(self, avg, parent_avg, volume):		
		norm = np.power(np.sum(np.power(np.abs(avg - parent_avg), self.power)) * volume, (1/self.power))
		return norm

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
		
		# INTERSECTION: [LEFT, RIGHT, DOWN, UP]		

		if base_node_id == 0:
			vals[:, base_node_id] = self.compute_average_score_from_tree(estimator.tree_.value[base_node_id])
			norms[base_node_id] = self.__compute_norm(vals[:, base_node_id], 0, 1)
			rectangles[base_node_id] = np.array([-2., 2., -2., 2.])			

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

	##
	def __variable_importance(self, estimator, base_node_id, vi_node_box, vi_norms, vi_vals, feature, threshod):        
		if base_node_id == 0:
			vi_vals[:, base_node_id] = estimator.tree_.value[base_node_id][:, 0]
			vi_norms[base_node_id] = self.__compute_norm(vi_vals[:, base_node_id], 0, 1)

		left_id = estimator.tree_.children_left[base_node_id]
		right_id = estimator.tree_.children_right[base_node_id]
		if left_id >= 0:
			vi_node_box[left_id, :, :] = vi_node_box[base_node_id, :, :]
			vi_node_box[left_id, estimator.tree_.feature[base_node_id], 1] = np.min(
				[estimator.tree_.threshold[base_node_id],
				 vi_node_box[left_id, estimator.tree_.feature[base_node_id], 1]])
			self.__variable_importance(estimator, left_id, vi_node_box, vi_norms, vi_vals, feature, self.vi_threshold)
			vi_vals[:, left_id] = estimator.tree_.value[left_id][:, 0] - estimator.tree_.value[base_node_id][:, 0]
			tnorm = self.__compute_norm(vi_vals[:, left_id], vi_vals[:, base_node_id], 1)
			if estimator.tree_.feature[estimator.tree_.children_left[base_node_id]] == feature and tnorm > threshod:
				vi_norms[left_id] = tnorm
		if right_id >= 0:
			vi_node_box[right_id, :, :] = vi_node_box[base_node_id, :, :]
			vi_node_box[right_id, estimator.tree_.feature[base_node_id], 0] = np.max(
				[estimator.tree_.threshold[base_node_id],
				 vi_node_box[right_id, estimator.tree_.feature[base_node_id], 0]])
			self.__variable_importance(estimator, right_id, vi_node_box, vi_norms, vi_vals, feature, self.vi_threshold)
			vi_vals[:, right_id] = estimator.tree_.value[right_id][:, 0] - estimator.tree_.value[base_node_id][:, 0]
			tnorm = self.__compute_norm(vi_vals[:, right_id], vi_vals[:, base_node_id], 1)
			if estimator.tree_.feature[estimator.tree_.children_right[base_node_id]] == feature and tnorm > threshod:
				vi_norms[right_id] = tnorm

	##

	def predict(self, X, m=10, start_m=0, paths=None):
		'''
		Predict using a maximum of M-terms
		:X: Data samples.
		:m: Maximum of M-terms.
		:start_m: The index of the starting term. Can be used to evaluate predictions incrementally over terms.paths.shape
		:paths: Instead of computing decision paths for each sample, the method can receive the indicator matrix. Can be used to evaluate predictions incrementally over terms.
		:return: Predictions.
		'''
		sorted_norms = np.argsort(-self.norms)[start_m:m]
		if paths == None:
			paths, n_nodes_ptr = self.rf.decision_path(X)
		pruned = lil_matrix(paths.shape, dtype=np.float32)
		
		pruned[:, sorted_norms] = paths[:, sorted_norms]
		predictions = pruned * self.vals.T / len(self.rf.estimators_)
		return predictions

	def evaluate_angle_smoothness(self, m=1000, error_TH=0, text='', output_folder=''):
		'''
		Evaluate smoothness using sparsity consideration
		'''
		approx_diff = False		
		mask = np.ones(len(self.norms), dtype=bool)
		mask[self.root_nodes] = False
		norms = self.norms[mask]

		p = 2
		h = 0.01
		diffs = []		
		taus = np.arange(1., 10., h)		
		total_sparsities, total_alphas = [], []
		for tau in tqdm(taus):
			tau_sparsity = np.power(np.power(norms, tau).sum(), ((1/tau)-1))
			tau_sparsity *= np.power(norms, (tau-1)).sum()
			diffs.append(tau_sparsity)				
		diffs = -np.array(diffs)

		angles = np.rad2deg(np.arctan(diffs))
		
		plt.figure(1)		
		plt.title(f"tau vs. angle")
		plt.xlabel(f'tau')
		plt.ylabel(f'sparsity angle')
		plt.plot(taus, angles, zorder=1)

		save_path = os.path.join(output_folder, f"{text}_derivates.png")
		print(f"save_path:{save_path}")
		plt.savefig(save_path, \
		    dpi=300, bbox_inches='tight')
		plt.clf()

		plt.figure(2)
		plt.title(f"tau vs. derivative")
		plt.xlabel(f'tau')
		plt.ylabel(f'sparsity derivative')
		plt.plot(taus, diffs, zorder=1)

		save_path = os.path.join(output_folder, f"{text}_angles.png")
		print(f"save_path:{save_path}")
		plt.savefig(save_path, \
		    dpi=300, bbox_inches='tight')
		plt.clf()		

		print(f"abs(angles+90.).min():{abs(angles+90.).min()}")
		epsilon_1 = 0.8
		epsilon_2 = 3*epsilon_1
		
		epsilon_1_indices = np.where(abs(angles+90.)<=epsilon_1)[0]
		epsilon_2_indices = np.where(abs(angles+90.)<=epsilon_2)[0]
		
		angle_index_1 = epsilon_1_indices[-1]
		angle_index_2 = epsilon_2_indices[-1]

		critical_tau_approximation_1 = taus[angle_index_1]
		critical_alpha_approximation_1 = ((1/critical_tau_approximation_1) - 1/p)

		critical_tau_approximation_2 = taus[angle_index_2]
		critical_alpha_approximation_2 = ((1/critical_tau_approximation_2) - 1/p)

		return critical_alpha_approximation_1, critical_alpha_approximation_2
		

	def evaluate_smoothness(self, m=1000, error_TH=0.):
		'''
		Evaluates smoothness for a maximum of M-terms
		:m: Maximum terms to use. Default is 1000.
		:return: Smothness index, n_wavelets, errors.
		'''		
		n_wavelets = []
		errors = []
		step = 10
		power = 2
		print_errors = False		
		paths, n_nodes_ptr = self.rf.decision_path(self.X)
		predictions = np.zeros(np.shape(self.y))
		for m_step in range(2, m, step):
			if m_step > len(self.norms):
				break			
			predictions += self.predict(self.X, m=m_step, start_m=max(m_step - step, 0), paths=paths)

			import pdb; pdb.set_trace()
			error_norms = np.power(np.sum(np.power(self.y - predictions, power), 1), 1. / power)
			total_error = np.sum(np.square(error_norms), 0) / len(self.X)

			if len(errors)> 0:
				if errors[-1] == total_error:
					break


			if total_error < error_TH:
				break
			
			if m_step > 0 and total_error > 0:
				if print_errors:
					logging.info('Error for m=%s: %s' % (m_step - 1, total_error))
				n_wavelets.append(m_step - 1)
				errors.append(total_error)
		logging.info(f"total m_step is {m_step}")        


		plt.figure(0.5)        
		plt.clf()		
		n_wavelets = np.reshape(n_wavelets, (-1, 1))
		errors = np.reshape(errors, (-1, 1))
		plt.title(f'#wavelets to errors, DS size: {self.X.shape[0]}')
		plt.xlabel('#wavelets')
		plt.ylabel('errors')
		plt.plot(n_wavelets, errors)
		plt.draw()
		plt.pause(0.5)
		

		plt.figure(2)
		plt.clf()
		n_wavelets_log = np.log(np.reshape(n_wavelets, (-1, 1)))
		errors_log = np.log(np.reshape(errors, (-1, 1)))
		plt.title(f'log(#wavelets) to log(errors), DS size: {self.X.shape[0]}')
		plt.xlabel('log(#wavelets)')
		plt.ylabel('log(errors)')

		if self.save_errors:
			def convert(o):
				if isinstance(o, np.generic): return o.item()  
				raise TypeError

			dir_path = r"C:\projects\RFWFC\results\approximation_methods\RF_50_TREE"
			json_file_name = "50000_points_50_TREE.json"
			write_data = {}
			write_data['n_wavelets'] = list(n_wavelets.squeeze())
			write_data['errors'] = list(errors.squeeze())			
			# write_data['mean_norms'] = list(mean_norms)
			path = os.path.join(dir_path, json_file_name)
			with open(path, "w+") as f:
				json.dump(write_data, f, default=convert)
			print(f"saved errors to {path}")

		regr = linear_model.LinearRegression()
		# regr = linear_model.Ridge(alpha=.8)
		
		regr.fit(n_wavelets_log, errors_log)

		y_pred = regr.predict(n_wavelets_log)
		plt.plot(n_wavelets_log, y_pred, color='blue', linewidth=3)	

		alpha = np.abs(regr.coef_[0][0])

		# plt.plot(n_wavelets_log, errors_log, label=f'alpha:{alpha}')
		# plt.legend()
		# plt.draw()
		# plt.pause(1)		

		# logging.info('Smoothness index: %s' % alpha)

		return alpha, n_wavelets, errors


	def save_wavelet_norms(self):
		result = self.norms
		result[self.root_nodes] = 0.
		result = list(result)		
		# TODO: IMPLEMENT FOR MORE THAN ONE TREE!
		# remove root node
		result = result[1:]
		
		def convert(o):
			if isinstance(o, np.generic): return o.item()
			raise TypeError

		dir_path = r"C:\projects\RFWFC\results\approximation_methods\Sparsity\RF_5_TREE"
		json_file_name = "norms_50000.json"
		write_data = {}		
		write_data['norms'] = result
		write_data['num_trees'] = len(self.rf.estimators_)
		norms_path = os.path.join(dir_path, json_file_name)
		with open(norms_path, "w+") as f:
			json.dump(write_data, f, default=convert)

		print(f"saved norms to:{norms_path}")

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
	# X, y = make_classification(n_samples=1000, n_features=2,
	#                   n_informative=2, n_redundant=0,
	#                   random_state=0, shuffle=False)
	# clf = RandomForestClassifier(max_depth=2, random_state=0)
	# clf.fit(X, y)
	# ax = plt.gca()    
	from sklearn import tree
	X = [[0, 0], [1, 1]]
	Y = [0, 1]  
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(X, Y)
