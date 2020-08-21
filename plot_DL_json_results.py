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
for file_path in Path(main_dir).glob('**/*.json'):	
	file_path = str(file_path)	
	epoch = file_path.split('\\')[-2].split('.')[-2]
	eps = file_path.split('\\')[-2].split('.')[1]
	# if eps != '05_0':
	# 	continue
	# if int(epoch)%5 != 0:
	# 	continue
	with open(file_path, "r+") as f:
		result = json.load(f)	
	sizes = result["sizes"]
	alphas = result["alphas"]
	# plt.fill_between(sizes, [k[0] for k in alphas], [k[1] for k in alphas], \
	# 	alpha=0.2, linewidth=4)
	plt.plot(sizes, [np.array(k).mean()	 for k in alphas], label=f"{epoch}")

plt.legend()
plt.title("alphas for different epochs")
plt.xlabel(f'layer')
plt.ylabel(f'evaluate_smoothnes index- alpha')
plt.show()