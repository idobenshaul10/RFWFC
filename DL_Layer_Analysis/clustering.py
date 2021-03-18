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

def kmeans_cluster(X, Y, total_num_layers=-1, visualize=False, output_folder=None, layer_str="", \
		sample_size=3000, save_graph=False, fig=None):
	plt.clf()
	plt.figure(num=None, figsize=(8, 6))
	np.random.seed(2)
	k = len(np.unique(Y))
	print(f"Fitting k means with k={k}")
	kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
	save_path = os.path.join(output_folder, f"{layer_str}.p")
	if save_graph:
		pickle.dump(kmeans, open(save_path, "wb"))
	if not visualize:		
		return kmeans
	predicted_labels = kmeans.labels_
	print(f"Fitting umap")
	reducer = umap.UMAP(random_state=42)
	indices = np.random.choice(len(X), sample_size)

	X = X[indices]
	Y = Y[indices]
	predicted_labels = predicted_labels[indices]	
	
	embedding_train = reducer.fit_transform(X)
	print(f"Done fitting umap")	
	
	ax = fig.add_subplot(total_num_layers//2, (2+total_num_layers%2), int(layer_str)+2)
	scatter = ax.scatter(
		embedding_train[:, 0], embedding_train[:, 1], c=predicted_labels, cmap="Spectral" , s=0.5
	)
	legend1 = ax.legend(*scatter.legend_elements(),
			loc="lower left", title="Classes")

	plt.setp(ax, xticks=[], yticks=[])	
	plt.suptitle(f"UMAP of Clustering for {layer_str}", fontsize=18)
	ax.set_xlabel(f"layer:{layer_str}")

	if save_graph:
		save_path = os.path.join(output_folder, f"{layer_str}.png")
		print(f"save_path:{save_path}")
		plt.savefig(save_path, \
			dpi=300, bbox_inches='tight')
	
	return kmeans

def get_clustering_statistics(X, Y, kmeans):
	Y = Y.squeeze()	
	metrics_results = {}
	preds = kmeans.labels_
	print("START clustering statistics")
	metrics_results['silhouette_score'] = metrics.silhouette_score(X, preds)	
	metrics_results['adj_rand'] = metrics.adjusted_rand_score(Y, preds)
	metrics_results['MI_score'] = metrics.adjusted_mutual_info_score(Y, preds)
	metrics_results['homogeneity_score'] = metrics.homogeneity_score(Y, preds)
	metrics_results['completeness'] = metrics.completeness_score(Y, preds)
	metrics_results['FMI'] = metrics.fowlkes_mallows_score(Y, preds)
	print("DONE clustering statistics")
	for k, v in metrics_results.items():
		print(f'{k}:{v}')
	return metrics_results

if __name__ == '__main__':	

	from sklearn.datasets import make_blobs
	X, Y = make_blobs(n_samples=5000, centers=10, n_features=200, random_state=0)
	output_folder = r"C:\projects\RFWFC\results"
	kmeans = kmeans_cluster(X, Y, False, output_folder, layer_str="", sample_size=500)
	import pdb; pdb.set_trace()
	get_clustering_statistics(X, Y, kmeans)
