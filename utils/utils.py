import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold
import sys
from tree_models.random_forest import WaveletsForestRegressor
import torchvision

def normalize_data(x_raw):
	x = (x_raw - np.min(x_raw, 0))/(np.max(x_raw, 0) - np.min(x_raw, 0))
	x = np.nan_to_num(x)
	return x

def train_model(x, y, method='RF', mode='regression', trees=5, depth=9, features='auto',
				state=2000, threshold=1000, train_vi=False, nnormalization='volume', cube_length=1.):

	
	model = WaveletsForestRegressor(regressor='random_forest', \
		mode=mode, trees=trees, depth=depth, train_vi=train_vi, features=features, \
		seed=state, vi_threshold=threshold, norms_normalization=nnormalization, cube_length=cube_length)
	
	model.fit(x, y)
	return model


def run_alpha_smoothness(X, y, t_method='RF', n_trees=1, m_depth=9,
						 n_features='auto', n_state=2000, normalize=True, \
						 norm_normalization='volume', cube_length=1., error_TH=0.1, \
						 text='', output_folder='', epsilon_1=None, epsilon_2=None):
	
	if normalize:
		X = normalize_data(X)

	norm_m_term = 0    
	model = train_model(X, y, method=t_method, trees=n_trees, \
		depth=m_depth, features=n_features, state=n_state, \
		nnormalization=norm_normalization, cube_length=cube_length)
	

	alpha = model.evaluate_angle_smoothness(text=text, \
		output_folder=output_folder, epsilon_1=epsilon_1, epsilon_2=epsilon_2)
	n_wavelets, errors = 0., 0.

	logging.log(60, 'ALPHA SMOOTHNESS over X: ' + str(alpha))
	return alpha, -1, -1, norm_m_term, model


def save_wavelet_norms(X, y, t_method='RF', num_wavelets=1000, n_trees=1, m_depth=9,
						 n_features='auto', n_state=2000, normalize=True, \
						 norm_normalization='volume', cube_length=1.):
	if normalize:
		X = normalize_data(X)

	norm_m_term = 0
	model = train_model(X, y, method=t_method, trees=n_trees, \
		depth=m_depth, features=n_features, state=n_state, \
		nnormalization=norm_normalization, cube_length=cube_length)
	model.save_wavelet_norms()



def calculate_besov_semi_norm(X, y, t_method='RF', num_wavelets=1000, n_trees=1, m_depth=9,
						 n_features='auto', n_state=2000, normalize=True, \
						 norm_normalization='volume', cube_length=1.):
	if normalize:
		X = normalize_data(X)

	norm_m_term = 0
	model = train_model(X, y, method=t_method, trees=n_trees, \
		depth=m_depth, features=n_features, state=n_state, \
		nnormalization=norm_normalization, cube_length=cube_length)
	model.calculate_besov_semi_norm()


def imshow(img):
	# img = img / 2 + 0.5
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()    

def visualize_augmentation(images):    
	imshow(torchvision.utils.make_grid(images))
	import pdb; pdb.set_trace()

if __name__ == '__main__':    
	from sklearn.datasets import make_moons, make_circles, make_classification
	X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
						   random_state=1, n_clusters_per_class=1)

	import pdb; pdb.set_trace()