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
import albumentations as A
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from utils.utils import visualize_augmentation
from datetime import datetime
import importlib
from sklearn.model_selection import KFold
from shutil import copyfile
from pathlib import Path

# USAGE:  python .\train\train_mnist.py --output_path "C:\projects\RFWFC\results\trained_models\mnist\normal\" --batch_size 32 --epochs 100
parser = argparse.ArgumentParser(description='train lenet5 on mnist dataset')
parser.add_argument('--output_path', default=r"C:\projects\DL_Smoothness_Results\trained_models", 
	help='output_path for checkpoints')
parser.add_argument('--seed', default=0, type=int, help='seed')
parser.add_argument('--lr', default=0.001, type=float, help='lr for train')
parser.add_argument('--batch_size', default=32, type=int, help='batch_size for train/test')
parser.add_argument('--epochs', default=100, type=int, help='num epochs for train')
parser.add_argument('--num_classes', default=10, type=int, help='num categories in output of model')
parser.add_argument('--env_name', type=str, default="mnist")

parser.add_argument('--kfolds', default=5, type=int, help='number of folds for cross-validation')
parser.add_argument('--enrich_factor', default=1., type=float, help='num categories in output of model')
parser.add_argument('--enrich_dataset', action="store_true", help='if True, will show sample images from DS')
parser.add_argument('--visualize_dataset', action="store_true", help='if True, will show sample images from DS')


args, _ = parser.parse_known_args()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_SEED = args.seed
LEARNING_RATE = args.lr
BATCH_SIZE = args.batch_size
N_EPOCHS = args.epochs
N_CLASSES = args.num_classes
ENRICH_FACTOR = args.enrich_factor
softmax = nn.Softmax(dim=1)
torch.manual_seed(RANDOM_SEED)

m = '.'.join(['environments', args.env_name])
module = importlib.import_module(m)
dict_input = vars(args)
environment = eval(f"module.{args.env_name}()")

model, train_dataset, test_dataset, layers = environment.load_enviorment()
time_filename = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
output_path = os.path.join(args.output_path, f"{args.env_name}_{time_filename}")

if not os.path.isdir(output_path):
	os.mkdir(output_path)

path = Path(__file__)
model_path = os.path.join(path.parents[1], 'models', f"{type(model).__name__}.py")
copyfile(model_path, os.path.join(output_path, "model.py"))

AUG = A.Compose({
	A.Resize(32, 32),	
	# A.HorizontalFlip(p=0.5),
	A.Rotate(limit=(-25, 25)),
	# A.VerticalFlip(p=0.5),
	# A.Normalize((0.5), (0.5))
	A.OpticalDistortion()
})

def transform(image):		
	image = AUG(image=np.array(image))['image']		
	image = torch.tensor(image, dtype=torch.float)	
	return image

def enrich_dataset(dataset, factor=1.):
	new_dataset_size = int(len(dataset) * factor)		
	indices = np.random.choice(len(dataset), new_dataset_size)
	transformed_images = []
	for index in tqdm(indices):
		image = dataset[index][0]
		image = transform(image)/255.
		transformed_images.append(image.view(1, 1, 32, 32))

	labels = [dataset[i][1] for i in indices]		
	labels = torch.tensor(labels)		
	transformed_images = torch.cat(transformed_images)		
	new_dataset = TensorDataset(transformed_images, labels)
	return new_dataset

def train(train_loader, model, criterion, optimizer, device):
	model.train()
	running_loss = 0	
	for X, y_true in train_loader:
		optimizer.zero_grad()		
		X = X.to(device)
		y_true = y_true.to(device)
		y_hat = model(X)
		loss = criterion(y_hat, y_true.long())
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
		y_true = y_true.to(device).long()
		y_hat = model(X)
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
			y_true = y_true.to(device).long()		
			logits = model(X)
			probs = softmax(logits)
			predicted_labels = torch.max(probs, 1)[1]
			n += y_true.size(0)			
			correct_pred += (predicted_labels == y_true).sum()
	return correct_pred.float() / n

def save_epoch(output_path, epoch, fold_index, model, train_acc, valid_acc):
	fold_folder = os.path.join(output_path, str(fold_index))
	if not os.path.isdir(fold_folder):
		os.mkdir(fold_folder)
	checkpoint_path = os.path.join(fold_folder, f"weights.best.h5")
	model_state_dict = model.state_dict()
	state_dict = OrderedDict()
	state_dict["epoch"] = epoch
	state_dict["checkpoint"] = model_state_dict
	state_dict["train_acc"] = train_acc
	state_dict["valid_acc"] = valid_acc
	save(state_dict, checkpoint_path)

def training_loop(model, criterion, optimizer, train_loader, valid_loader, \
	epochs, device, print_every=1, fold_index=1):
	best_loss = 1e10
	train_losses = []
	valid_losses = []
	best_val_acc = -1 
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

			if valid_acc > best_val_acc:
				best_val_acc = valid_acc
				save_epoch(output_path, epoch, fold_index, model, train_acc, valid_acc)				
	
	return model, optimizer, (train_losses, valid_losses)

if args.enrich_dataset:	
	train_dataset = enrich_dataset(train_dataset, factor=ENRICH_FACTOR)
	if args.visualize_dataset:
		random_indices = np.random.choice(len(train_dataset), 16)
		to_show_images = []
		for i in random_indices:
			image = train_dataset[i][0]
			to_show_images.append(image)
		visualize_augmentation(to_show_images)


optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

if args.kfolds > 1:
	x_train = [x.unsqueeze(0) for x, y in train_dataset]
	y_train = [y for x, y in train_dataset]

	x_train = np.vstack(x_train)
	y_train = np.array(y_train)
	kfold =KFold(n_splits=args.kfolds)
	
	for fold_index, (train_index, test_index) in enumerate(kfold.split(x_train)):
		print(f"fold_index:{fold_index}")

		x_train_fold = torch.tensor(x_train[train_index])
		y_train_fold = torch.tensor(y_train[train_index])
		x_test_fold = torch.tensor(x_train[test_index])
		y_test_fold = torch.tensor(y_train[test_index])

		fold_train_dataset = TensorDataset(x_train_fold, y_train_fold)
		fold_val_dataset = TensorDataset(x_test_fold, y_test_fold)
		
		train_loader = DataLoader(dataset=fold_train_dataset, 
								  batch_size=BATCH_SIZE, 
								  shuffle=True)

		valid_loader = DataLoader(dataset=fold_val_dataset, 
								  batch_size=BATCH_SIZE, 
								  shuffle=False)
		
		model, optimizer, _ = training_loop(model, criterion, optimizer, \
			train_loader, valid_loader, N_EPOCHS, DEVICE, fold_index=fold_index)