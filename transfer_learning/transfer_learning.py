import seaborn as sns
# PyTorch
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import warnings
# Data science tools
import numpy as np
import pandas as pd
import os
from PIL import Image
from datasets.caletch101 import Caltech101
import matplotlib.pyplot as plt

# https://github.com/WillKoehrsen/pytorch_challenge/blob/master/Transfer%20Learning%20in%20PyTorch.ipynb
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/caltech.py

# plt.rcParams['font.size'] = 14

# datadir = r'C:\projects\DL_Smoothness_Results\transfer_learning'
# traindir = os.path.join(datadir, 'train')
# validdir = os.path.join(datadir,'valid')
# testdir = os.path.join(datadir, 'test/')

# save_file_name = 'vgg16-transfer-4.pt'
# checkpoint_path = 'vgg16-transfer-4.pth'
# batch_size = 128
# # Whether to train on a gpu
# train_on_gpu = cuda.is_available()
# print(f'Train on gpu: {train_on_gpu}')

# if train_on_gpu:
# 	gpu_count = cuda.device_count()
# 	print(f'{gpu_count} gpus detected.')
# 	if gpu_count > 1:
# 		multi_gpu = True
# 	else:
# 		multi_gpu = False


dataset = Caltech101()


