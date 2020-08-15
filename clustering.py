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

def kmeans_cluster(X, Y, output_folder=None, layer_str = ""):
	k = len(np.unique(Y))
	print(f"Fitting k means with k={k}")
	kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
	predicted_labels = kmeans.labels_
	print(f"Fitting umap")
	reducer = umap.UMAP(random_state=42)
	indices = np.random.choice(len(X), 5000)

	X = X[indices]
	Y = Y[indices]
	predicted_labels = predicted_labels[indices]	
	
	embedding_train = reducer.fit_transform(X)
	print(f"Done fitting umap")
	
	fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(12, 10))
	ax.scatter(
		embedding_train[:, 0], embedding_train[:, 1], c=predicted_labels, cmap="Spectral" , s=0.5
	)

	plt.setp(ax, xticks=[], yticks=[])	
	plt.suptitle(f"UMAP of Clustering for {layer_str}", fontsize=18)	
	ax.set_xlabel(f"clustering inertia:{kmeans.inertia_}")
	save_graph=True
	if save_graph:
		save_path = os.path.join(output_folder, f"{layer_str}.png")
		print(f"save_path:{save_path}")
		plt.savefig(save_path, \
			dpi=300, bbox_inches='tight')

	plt.clf()

if __name__ == '__main__':
	N = 20
	X = np.random.rand(N, 20)
	Y = np.random.randint(0, 10, (N, 1)) 
	kmeans_cluster(X, Y)
