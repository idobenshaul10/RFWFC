import numpy as np
import pandas as pd
import os
from sklearn import tree, linear_model, ensemble
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix, lil_matrix
import operator
import code
from functools import reduce
import argparse
import logging
from alpha_index_plotter import *
from random_forest import WaveletsForestRegressor

def load_csv(file_path):
	return pd.read_csv(file_path, delimiter=',', header=None).values

def load_npz(file_path, name):
	return np.load(file_path)[name]

class LoadFromFile (argparse.Action):
	def __call__ (self, parser, namespace, values, option_string = None):
		with open(values) as f:
			parsed = parser.parse_args(f.read().split(), namespace)
			return parsed

def main():
	parser = argparse.ArgumentParser(description='WaveletsForestRegressor runner. Use "python -m pydoc random_forest" or see "random_forest.html" for more details.')
	default_config_path = 'config.txt'
	config_action = parser.add_argument('--config', default=default_config_path, action=LoadFromFile)
	parser.add_argument('--log', default='INFO',help='Logging level. Default is INFO.')
	parser.add_argument('--regressor',default='WF',help='Regressor type.')
	parser.add_argument('--trees',default=1,type=int,help='Number of trees in the forest.')
	parser.add_argument('--features',default='auto',help='Features to consider in each split. Same options as sklearn\'s DecisionTreeRegressor.')
	parser.add_argument('--depth', default=9,type=int,help='Maximum depth of each tree.Use 0 for unlimited depth.')
	parser.add_argument('--seed',default=2000,type=int,help='Seed')
	parser.add_argument('--criterion',default='gini',help='Splitting criterion.')
	parser.add_argument('--bagging',default=0.8,type=float,help='Bagging. Only available when using the "decision_tree_with_bagging" regressor.')
	# parser.add_argument('--data',default='trainingData.csv',help='Training data path. Default is "trainingData.csv".')
	# parser.add_argument('--labels',default='trainingLabel.csv',	help='Training labels path. Default is "trainingLabel.csv".')	
	
	parser.add_argument('--output_path',default=None,help='Splitting criterion.')
	parser.add_argument('--dimension',default=2,type=int, help='Dimension for sphere in R^n experiment')
	# parser.add_argument('--dataset_size',default=5000,type=int,	help='number of points to draw from unit_box(for more info- check PointGenerator')	


	flags, _ = parser.parse_known_args()
	if os.path.exists(flags.config):
			config_flags = config_action(parser, argparse.Namespace(), flags.config)
			aux_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
			for arg in vars(flags): aux_parser.add_argument('--'+arg)
			flags, _ = aux_parser.parse_known_args(namespace=config_flags)      

	np.random.seed(flags.seed)
	logging.basicConfig(level=getattr(logging, flags.log))
	logging.info('Creating regressor with (dimension=%s, regressor=%s, trees=%s, features=%s, depth=%s, seed=%s, criterion=%s, bagging=%s)' % (flags.dimension, flags.regressor, flags.trees, flags.features, flags.depth, flags.seed, flags.criterion, flags.bagging) )
	
	data_str = f'Dimension:{flags.dimension}, # Trees:{flags.trees}, Depth:{flags.depth}'
	plot_alpha_per_num_sample_points(flags, data_str, output_path=flags.output_path)
	
	# plot_alpha_per_depth(flags, \
	# 		data_str, output_path=flags.output_path)

	# plot_alpha_per_tree_number(flags, \
	# 		data_str, output_path=flags.output_path)
	

if '__main__' == __name__:
		main()