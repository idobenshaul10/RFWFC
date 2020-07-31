import numpy as np
from datetime import datetime
import torch
from torch import save
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os 
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from models.LeNet5 import LeNet5
import argparse
from collections import OrderedDict

parser = argparse.ArgumentParser(description='train lenet5 on mnist dataset')
parser.add_argument('--output_path', default=r"C:\projects\RFWFC\results\DL_layers\trained_models", 
	help='output_path for checkpoints')
args, _ = parser.parse_known_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 32
N_EPOCHS = 100
IMG_SIZE = 32
N_CLASSES = 10
output_path = os.path.join(args.output_path, "LeNet5")
if not os.path.isdir(output_path):
	os.mkdir(output_path)

def train(train_loader, model, criterion, optimizer, device):
	model.train()
	running_loss = 0	
	for X, y_true in train_loader:
		optimizer.zero_grad()		
		X = X.to(device)
		y_true = y_true.to(device)
		y_hat, _ = model(X) 
		loss = criterion(y_hat, y_true) 
		running_loss += loss.item() * X.size(0)		
		loss.backward()
		optimizer.step()
		
	epoch_loss = running_loss / len(train_loader.dataset)
	return model, optimizer, epoch_loss


def validate(valid_loader, model, criterion, device):	
	model.eval()
	running_loss = 0
	
	for X, y_true in valid_loader:
	
		X = X.to(device)
		y_true = y_true.to(device)
		y_hat, _ = model(X)
		loss = criterion(y_hat, y_true) 
		running_loss += loss.item() * X.size(0)

	epoch_loss = running_loss / len(valid_loader.dataset)
	return model, epoch_loss

def get_accuracy(model, data_loader, device):	
	correct_pred = 0
	n = 0
	with torch.no_grad():
		model.eval()
		for X, y_true in data_loader:
			X = X.to(device)
			y_true = y_true.to(device)
			_, y_prob = model(X)
			_, predicted_labels = torch.max(y_prob, 1)
			n += y_true.size(0)
			correct_pred += (predicted_labels == y_true).sum()
	return correct_pred.float() / n

def training_loop(model, criterion, optimizer, train_loader, valid_loader, \
		epochs, device, print_every=1, save_every=5):
	best_loss = 1e10
	train_losses = []
	valid_losses = [] 
	for epoch in range(0, epochs):

		model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
		train_losses.append(train_loss)

		with torch.no_grad():
			model, valid_loss = validate(valid_loader, model, criterion, device)
			valid_losses.append(valid_loss)

		if epoch % print_every == (print_every - 1):
			
			train_acc = get_accuracy(model, train_loader, device=device)
			valid_acc = get_accuracy(model, valid_loader, device=device)
				
			print(f'{datetime.now().time().replace(microsecond=0)} --- '
				  f'Epoch: {epoch}\t'
				  f'Train loss: {train_loss:.4f}\t'
				  f'Valid loss: {valid_loss:.4f}\t'
				  f'Train accuracy: {100 * train_acc:.2f}\t'
				  f'Valid accuracy: {100 * valid_acc:.2f}')

		if epoch % save_every == 0:			
			checkpoint_path = f"{output_path}/weights.{epoch}.h5"
			model_state_dict = model.state_dict()
			state_dict = OrderedDict()
			state_dict["epoch"] = epoch
			state_dict["checkpoint"] = model_state_dict
			state_dict["train_acc"] = train_acc
			state_dict["valid_acc"] = valid_acc
			save(state_dict, checkpoint_path)
	
	return model, optimizer, (train_losses, valid_losses)


# define transforms
transforms = transforms.Compose([transforms.Resize((32, 32)),
								 transforms.ToTensor()])

# download and create datasets
train_dataset = datasets.MNIST(root=r'C:\datasets\mnist_data', 
							   train=True, 
							   transform=transforms,
							   download=True)

valid_dataset = datasets.MNIST(root=r'C:\datasets\mnist_data', 
							   train=False, 
							   transform=transforms)

# define the data loaders
train_loader = DataLoader(dataset=train_dataset, 
						  batch_size=BATCH_SIZE, 
						  shuffle=True)

valid_loader = DataLoader(dataset=valid_dataset, 
						  batch_size=BATCH_SIZE, 
						  shuffle=False)


torch.manual_seed(RANDOM_SEED)
model = LeNet5(N_CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, valid_loader, N_EPOCHS, DEVICE)