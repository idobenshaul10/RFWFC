import numpy as np
from datetime import datetime
import torch
from torch import save
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os 
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from models.LeNet5 import LeNet5
from models.fashion_mnist_model import FashionCNN
import argparse
from collections import OrderedDict
import albumentations as A
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from utils.utils import visualize_augmentation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description='train lenet5 on mnist dataset')
parser.add_argument('--output_path', default=r"C:\projects\RFWFC\results\DL_layers\trained_models", 
	help='output_path for checkpoints')
parser.add_argument('--seed', default=0, type=int, help='seed')
parser.add_argument('--lr', default=0.001, type=float, help='lr for train')
parser.add_argument('--batch_size', default=32, type=int, help='batch_size for train/test')
parser.add_argument('--epochs', default=100, type=int, help='num epochs for train')
parser.add_argument('--num_classes', default=10, type=int, help='num categories in output of model')

args, _ = parser.parse_known_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = FashionCNN().to(device)

transforms = transforms.Compose([transforms.Resize((28, 28)),
								 transforms.ToTensor()])

error = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
train_dataset = datasets.FashionMNIST(
	root = r'C:\datasets\fashion_mnist',
	train = True,
	download = True,
	transform = transforms)


valid_dataset = datasets.FashionMNIST(
	root = r'C:\datasets\fashion_mnist',
	train = False,
	download = True,
	transform = transforms)

train_loader = DataLoader(dataset=train_dataset, 
	batch_size=args.batch_size, shuffle=True)

test_loader = DataLoader(dataset=valid_dataset, 
	batch_size=args.batch_size, shuffle=False)

count = 0
save_every = 10

loss_list = []
iteration_list = []
accuracy_list = []
predictions_list = []
labels_list = []

for epoch in range(args.epochs):
	for images, labels in train_loader:
		images, labels = images.to(device), labels.to(device)
		import pdb; pdb.set_trace()
		train = Variable(images.view(args.batch_size, 1, 28, 28))
		labels = Variable(labels)
		outputs = model(train)
		loss = error(outputs, labels)        
		optimizer.zero_grad()        
		loss.backward()        
		optimizer.step()    
		count += 1	
	
	
	total = 0
	correct = 0
	for images, labels in test_loader:
		images, labels = images.to(device), labels.to(device)
		labels_list.append(labels)
	
		test = Variable(images.view(args.batch_size, 1, 28, 28))
	
		outputs = model(test)
	
		predictions = torch.max(outputs, 1)[1].to(device)
		predictions_list.append(predictions)
		correct += (predictions == labels).sum()
	
		total += len(labels)
	
	accuracy = correct * 100 / total	
	print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))

	if epoch % save_every == 0:
		if args.continue_path is not None:
			checkpoint_path = f"{output_path}/AFTER_LOAD_weights.{epoch}.h5"
		else:
			checkpoint_path = f"{output_path}/weights.{epoch}.h5"

		model_state_dict = model.state_dict()
		state_dict = OrderedDict()
		state_dict["epoch"] = epoch
		state_dict["checkpoint"] = model_state_dict
		state_dict["train_acc"] = train_acc
		state_dict["valid_acc"] = accuracy
		save(state_dict, checkpoint_path)


