from __future__ import print_function
import os 
import sys
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from torch.optim.lr_scheduler import StepLR
from sklearn import tree, linear_model, ensemble
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, ion, show
import numpy as np
import importlib
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from utils import *
import time
import json
import umap

def kmeans_cluster(X, Y, output_folder=None):	
	k = len(np.unique(Y))
	print(f"Fitting k means with k={k}")
	kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
	predicted_labels = kmeans.labels_	
	print(f"Fitting umap")

	reducer = umap.UMAP(random_state=42)
	embedding_train = reducer.fit_transform(X)
	print(f"Done fitting umap")	
	
	fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 10))
	ax[0].scatter(
    	embedding_train[:, 0], embedding_train[:, 1], c=predicted_labels, cmap="Spectral"  # , s=0.1
	)

	plt.setp(ax[0], xticks=[], yticks=[])
	plt.setp(ax[1], xticks=[], yticks=[])
	plt.suptitle("MNIST data embedded into two dimensions by UMAP", fontsize=18)
	ax[0].set_title("Training Set", fontsize=12)
	ax[1].set_title("Test Set", fontsize=12)
	plt.show()

def script():
	import matplotlib.pyplot as plt
	import seaborn as sns
	from sklearn.datasets import fetch_openml
	from sklearn.model_selection import train_test_split

	import umap

	sns.set(context="paper", style="white")

	mnist = fetch_openml("mnist_784", version=1)
	X_train, X_test, y_train, y_test = train_test_split(
	    mnist.data, mnist.target, stratify=mnist.target, random_state=42
	)

	reducer = umap.UMAP(random_state=42)
	print("here1")
	embedding_train = reducer.fit_transform(X_train)
	embedding_test = reducer.transform(X_test)
	print("here2")

	fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 10))
	ax[0].scatter(
	    embedding_train[:, 0], embedding_train[:, 1], c=y_train, cmap="Spectral"  # , s=0.1
	)
	ax[1].scatter(
	    embedding_test[:, 0], embedding_test[:, 1], c=y_test, cmap="Spectral"  # , s=0.1
	)
	plt.setp(ax[0], xticks=[], yticks=[])
	plt.setp(ax[1], xticks=[], yticks=[])
	plt.suptitle("MNIST data embedded into two dimensions by UMAP", fontsize=18)
	ax[0].set_title("Training Set", fontsize=12)
	ax[1].set_title("Test Set", fontsize=12)
	plt.show()
		
if __name__ == '__main__':
	# N = 20
	# X = np.random.rand(N, 20)
	# Y = np.random.randint(0, 10, (N, 1)) 
	# kmeans_cluster(X, Y)
	script()