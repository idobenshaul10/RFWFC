import numpy as np
from sklearn import tree, linear_model, ensemble
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix, lil_matrix
import operator
import code
from tqdm import tqdm
from functools import reduce
from binarytree import Node

class Rectangle:	
	# [LEFT, RIGHT, DOWN, UP]
	def __init__(self, left, right, down, up):
		self.data = (left, right, down, up)

	def get_volume(self):
		return (self.right - self.left) * (self.up - self.down)

class Node:
	def __init__(self, data, level=0):
		self.left = None
		self.right = None
		self.rect = Rectangle(data)
		self.level = level

	def PrintTree(self):
		print(self.data)

	def point_in_rect(self, point):
		x, y = point
		if x > self.rect[1] or x < self.rect[0]:
			return False
		if y > self.rect[3] or y < self.rect[2]:
			return False
		return True

	def indices_in_rectangle(self, X):
		indices = np.where(point_in_rect(self, X))
		return indices

class DyadicDecisionTreeModel:
	def __init__(self, depth=9, seed=None):    
		self.seed = seed
		self.random_state = np.random.RandomState(seed=self.seed)
		self.estimators_ = []
		self.cube_length = cube_length
		
		rectangle = Rectangle(-self.cube_length/2, self.cube_length/2, \
			-self.cube_length/2, self.cube_length/2)

		self.root = Node(rectangle)

	def fit(self, X_all, parent=self.root):
		import pdb; pdb.set_trace()

		indices = np.arange(len(X_all))
		random_state = np.random.RandomState(seed=self.seed)

		if len(indices) < 1:
			return

		p_left, p_right, p_down, p_up = parent.data		
		
		if parent.level % 2 == 0:			
			left_rectangle = Rectangle(p_left, (p_right+p_left)/2, p_down, p_up)
			right_rectangle = Rectangle((p_right+p_left)/2, p_right, p_down, p_up)
			
			left_node = Node(left_rectangle)
			self.root.left = left_node
			X_left_indices = left_node.indices_in_rectangle(X_all)
			self.fit(X_all[X_left_indices], self.root.left)

			right_node = Node(right_rectangle)
			self.root.right = right_node
			X_right_indices = right_node.indices_in_rectangle(X_all)
			self.fit(X_all[X_right_indices], self.root.right)
			

 #  def decision_path(self, X):
	# paths = []
	# for i in range(self.trees):
	#   current_paths = self.estimators_[i].decision_path(X)      
	#   paths = np.append(paths, current_paths)

	# paths_csr_dim = reduce(lambda s, x: s+x.shape[1], paths, 0)
	# paths_csr = lil_matrix((np.shape(X)[0], paths_csr_dim),dtype=np.float32)
	# current_i = 0
	# for current_paths in paths:
	#   paths_csr[:, current_i:current_i + current_paths.shape[1]] = current_paths
	#   current_i += current_paths.shape[1]

	# return paths_csr.tocsr(), []
  




