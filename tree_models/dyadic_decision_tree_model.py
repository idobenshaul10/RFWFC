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

class Rectangle:	
	# [LEFT, RIGHT, DOWN, UP]
	def __init__(self, left, right, down, up):
		self.data = (left, right, down, up)
		self.volume = (right - left) * (up - down)

	def points(self):
		return self.data	

class Node:
	def __init__(self, data, level, X_all=None, parent_indices=None):
		self.rect = data		
		self.level = level		
		self.id = -1

		if parent_indices is not None:
			self.indices = self.indices_in_rectangle(X_all, parent_indices)
			self.num_samples = self.indices.sum()

	def PrintTree(self):
		print(self.data)

	def indices_in_rectangle(self, X_all, parent_indices=None):		
		x, y = X_all[:, 0], X_all[:, 1]
		rect_points = self.rect.points()
		result_x = np.logical_and(x < rect_points[1] , x > rect_points[0])
		result_y = np.logical_and(y < rect_points[3] , y > rect_points[2])
		result = np.logical_and(result_x, result_y)
		if parent_indices is not None:
			result = np.logical_and(result, parent_indices)		
		return result

class DyadicDecisionTreeModel:
	def __init__(self, depth=9, seed=2000, verbose=False, cube_length=1.):    
		self.seed = seed		
		self.random_state = np.random.RandomState(seed=self.seed)
		self.estimators_ = []
		self.cube_length = cube_length
		self.nodes = []
		self.verbose = verbose
		self.non_zero_norm_indices = None
		
		rectangle = Rectangle(-self.cube_length/2, self.cube_length/2, \
			-self.cube_length/2, self.cube_length/2)		
		self.root = Node(rectangle, level=0)
		self.nodes.append(self.root)

	def add_dataset(self, X_all, y_all):
		self.X_all = X_all
		self.y_all = y_all.squeeze()

	def get_mean_value(self, node_id):
		node = self.nodes[node_id]		
		result = np.dot(self.y_all, node.indices)/ node.num_samples
		return result
 
	def init_ids(self):
		for idx, node in enumerate(self.nodes):
			node.id = idx
		# if self.verbose:
		# 	print(f"ids:{[k.id for k in self.nodes]}")

	def fit(self, X_all, parent=None, indices=None):
		if parent is None:
			parent = self.root			
			self.root.indices = np.ones(len(X_all)).astype(np.int)
			self.root.num_samples = self.root.indices.sum()			

		parent_indices = parent.indices
		new_level = parent.level + 1
		random_state = np.random.RandomState(seed=self.seed)
		
		if parent_indices.sum() < 5:	
			return

		p_left, p_right, p_down, p_up = parent.rect.points()

		if parent.level % 2 == 0:
			left_rectangle = Rectangle(p_left, (p_right+p_left)/2, p_down, p_up)
			right_rectangle = Rectangle((p_right+p_left)/2, p_right, p_down, p_up)
		else:
			left_rectangle = Rectangle(p_left, p_right, (p_up+p_down)/2, p_up)
			right_rectangle = Rectangle(p_left, p_right, p_down, (p_up+p_down)/2)
		
		left_node = Node(left_rectangle, level=new_level, \
			X_all=X_all, parent_indices=parent_indices)		

		if left_node.num_samples >= 5:
			parent.left = left_node	
			self.fit(X_all, parent=parent.left)		
			self.nodes.append(left_node)

		right_node = Node(right_rectangle, level=new_level, \
			X_all=X_all, parent_indices=parent_indices)		
		
		if right_node.num_samples >= 5:
			parent.right = right_node			
			self.fit(X_all, parent=parent.right)		
			self.nodes.append(right_node)
			

	def decision_path(self, X):
		decision_paths = np.zeros((X.shape[0], len(self.nodes)))		
		for idx, node in tqdm(enumerate(self.nodes), total=len(self.nodes)):			
			if self.non_zero_norm_indices[idx] == 0:
				continue
			points_in_node = node.indices_in_rectangle(X)
			decision_paths[:, idx] = points_in_node
		return decision_paths

	def print_tree(self):
		for idx, node in enumerate(self.nodes):
			mean_val = self.get_mean_value(node.id)			
			print(f"Node:{idx}, id:{node.id}, level:{node.level}, mean_val:{mean_val:.4f}, \
				indices:{node.num_samples} ")

	def test_tree_indices(self):
		max_level = np.max([k.level for k in self.nodes])
		levels = np.zeros(max_level + 1)		

		for idx, node in enumerate(self.nodes):
			levels[node.level] += node.indices.sum()

		# this will only work if there are the same number of levels for every subtree
		# todo - add check with leaves
		print(f"levels: {levels}")

