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

main_dir = sys.argv[1]
# paths = Path('src').rglob('*.json')
# indices = [5, 10, 20, 30]
fig, axes = plt.subplots(1, 2)
for file_path in Path(main_dir).glob('**/*.json'):	
	file_path = str(file_path)	
	epoch = file_path.split('\\')[-2].split('.')[-2]
	eps = file_path.split('\\')[-2].split('.')[1]	
	with open(file_path, "r+") as f:
		result = json.load(f)	
	sizes = result["sizes"]
	alphas = result["alphas"]

	test_stats = None
	if 'test_stats' in result:
		test_stats = result['test_stats']		
	# plt.fill_between(sizes, [k[0] for k in alphas], [k[1] for k in alphas], \
	# 	alpha=0.2, linewidth=4)
	axes[0].plot(sizes, [np.array(k).mean()	 for k in alphas], label=f"{epoch}")
	if test_stats is not None:
		axes[1].scatter(epoch, [test_stats['top_1_accuracy']], label=f"{epoch}")

plt.legend()
plt.title("alphas for different epochs")
plt.xlabel(f'layer')
plt.ylabel(f'evaluate_smoothnes index- alpha')
plt.show()