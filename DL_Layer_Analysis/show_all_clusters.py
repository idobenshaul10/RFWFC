from __future__ import print_function
import os 
import sys
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, ion, show
import numpy as np
import importlib
import os,sys,inspect
from pathlib import Path
import json
import cv2
import glob
# USAGE: python .\DL_Layer_Analysis\plot_DL_json_results.py --main_dir C:\projects\RFWFC\results\mnist

def get_args():
	parser = argparse.ArgumentParser(description='Show all clusters')	
	parser.add_argument('--main_dir', type=str ,help='clusters folder')
	args = parser.parse_args()
	return args

def plot_clusters(main_dir):	
	fig=plt.figure()
	file_paths = glob.glob(os.path.join(main_dir, "*.png"))		
	file_paths = [k for k in file_paths if '_' not in k.split('\\')[-1]]
	width, height =224, 224
	rows = 1
	cols = len(file_paths)
	axes=[]

	for idx, file in enumerate(file_paths):	    
	    b = cv2.imread(file)	    
	    axes.append( fig.add_subplot(rows, cols, idx+1) )
	    # subplot_title=("Subplot"+str(idx))
	    # axes[-1].set_title(subplot_title)  
	    plt.imshow(b)
	fig.tight_layout()    
	plt.show()

if __name__ == '__main__':
	args = get_args()	
	plot_clusters(args.main_dir)