from __future__ import print_function
import os 
import sys
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from torch.optim.lr_scheduler import StepLR
from sklearn import tree, linear_model, ensemble
import sklearn.metrics as metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, ion, show
import importlib
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from utils import *
import time
import json
import umap
import pickle
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

np.random.seed(2)

def get_dim_reduction_UMAP(X, output_dimension=1000):	
	print(f"computing umap")	
	t0 = time.time()
	reducer = umap.UMAP(random_state=42, n_components=output_dimension)	
	t1 = time.time()	
	embedding_train = reducer.fit_transform(X)
	print(f"umap took {t1-t0}", flush=True)	
	return embedding_train


def get_dim_reduction(X, output_dimension=1000):
	print(f"computing PCA")
	t0 = time.time()

	pca = PCA(n_components=output_dimension)

	_embedded = pca.fit_transform(X)
	t1 = time.time()
	print(f"PCA took {t1-t0}", flush=True)	
	return _embedded

def get_dim_reduction_truncSVD(X, output_dimension=1000):
	print(f"computing TruncatedSVD")	
	t0 = time.time()

	truncatedSVD = TruncatedSVD(n_components=output_dimension)
	_embedded = truncatedSVD.fit_transform(X)

	t1 = time.time()
	print(f"TruncatedSVD took {t1-t0}", flush=True)	
	return _embedded




