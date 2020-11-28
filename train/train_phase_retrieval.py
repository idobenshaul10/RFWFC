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
import pickle
import cv2

# USAGE:  python .\train\train_mnist.py --output_path "C:\projects\RFWFC\results\trained_models\mnist\normal\" --batch_size 32 --epochs 100
parser = argparse.ArgumentParser(description='train lenet5 on mnist dataset')
parser.add_argument('--output_path', default=r"C:\projects\DL_Smoothness_Results\trained_models", 
	help='output_path for checkpoints')
parser.add_argument('--seed', default=0, type=int, help='seed')
parser.add_argument('--lr', default=0.001, type=float, help='lr for train')
parser.add_argument('--batch_size', default=32, type=int, help='batch_size for train/test')
parser.add_argument('--epochs', default=100, type=int, help='num epochs for train')
parser.add_argument('--env_name', type=str, default="mnist")
parser.add_argument('--kfolds', default=5, type=int, help='number of folds for cross-validation')
parser.add_argument('--enrich_factor', default=1., type=float, help='num categories in output of model')
parser.add_argument('--enrich_dataset', action="store_true", help='if True, will show sample images from DS')
parser.add_argument('--visualize_dataset', action="store_true", help='if True, will show sample images from DS')
parser.add_argument('--use_residual', action="store_true")

args, _ = parser.parse_known_args()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_SEED = args.seed
LEARNING_RATE = args.lr
BATCH_SIZE = args.batch_size
N_EPOCHS = args.epochs
ENRICH_FACTOR = args.enrich_factor
softmax = nn.Softmax(dim=1)
torch.manual_seed(RANDOM_SEED)

m = '.'.join(['environments', args.env_name])
module = importlib.import_module(m)
dict_input = vars(args)
environment = eval(f"module.{args.env_name}()")

model, train_dataset, test_dataset, layers = environment.load_enviorment(**dict_input)
time_filename = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
output_path = os.path.join(args.output_path, f"{args.env_name}_{time_filename}")

if not os.path.isdir(output_path):
	os.mkdir(output_path)

path = Path(__file__)
model_path = os.path.join(path.parents[1], 'models', f"{type(model).__name__}.py")
copyfile(model_path, os.path.join(output_path, "model.py"))
print(f"Begining train, args:{args}")
pickle.dump(args, open(os.path.join(output_path, "args.p"), "wb"))

def get_image_from_phase(mag, dft):
	dft = dft.detach().cpu().numpy().astype(np.float32())	
	real, imag = dft[:,:,0], dft[:,:,1]	
	back = cv2.merge([real, imag])	
	img_back = cv2.idft(back)
	img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

	return img_back

def train(train_loader, model, criterion, optimizer, device, epoch):
	model.train()
	running_loss = 0
	for idx, (mag, dft, gt_img, label) in enumerate(tqdm(train_loader, total=len(train_loader))):
		optimizer.zero_grad()
		mag = mag.to(device)
		dft = dft.to(device)
		pred, img_back = model(mag)		
		label = label.to(device).long()
		plt.clf()

		if epoch % 3 == 0 and idx == 0:	
			img_back = img_back[0].squeeze().detach().cpu().numpy()
			# img_back = get_image_from_phase(mag[0], dft_pred[0])
			# gt_img = get_image_from_phase(mag[0], phase[0])			
			ax1 = plt.subplot(1,2,1)	
			ax1.imshow(gt_img[0], cmap='gray')
			ax2 = plt.subplot(1,2,2)
			ax2.imshow(img_back, cmap='gray')
			plt.pause(0.01)
		loss = criterion(pred, label)

		running_loss += loss.item() * mag.size(0)
		loss.backward()
		optimizer.step()
		
	epoch_loss = running_loss / len(train_loader.dataset)
	return model, optimizer, epoch_loss

def validate(valid_loader, model, criterion, device):	
	model.eval()
	running_loss = 0	
	for mag, phase, _, label in valid_loader:
		mag = mag.to(device)
		label = label.to(device).long()
		y_hat, _ = model(mag)
		loss = criterion(y_hat, label)
		running_loss += loss.item() * mag.size(0)		

	epoch_loss = running_loss / len(valid_loader.dataset)	
	return model, epoch_loss

def get_accuracy(model, data_loader, device):	
	correct_pred = 0
	n = 0
	with torch.no_grad():
		model.eval()
		for mag, phase, _, label in data_loader:
			mag = mag.to(device)			
			label = label.to(device).long()		
			logits, _ = model(mag)
			probs = softmax(logits)
			predicted_labels = torch.max(probs, 1)[1]
			n += label.size(0)
			correct_pred += (predicted_labels == label).sum()
	
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
	best_val_acc = float('inf')
	for epoch in range(0, epochs):		
		model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device, epoch)
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

			if valid_acc < best_val_acc:
				best_val_acc = valid_acc
				save_epoch(output_path, epoch, fold_index, model, train_acc, valid_acc)	

	
	return model, optimizer, (train_losses, valid_losses)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()

if args.kfolds > 1:
	x_train = [x.unsqueeze(0) for x, phase, _, _ in train_dataset]
	phase_train = [phase.unsqueeze(0) for x, phase, _, _ in train_dataset]	
	images = [img for x, phase, img, _ in train_dataset]
	labels = [label for _, _, _, label in train_dataset]
	
	x_train = np.vstack(x_train)
	phase_train = np.vstack(phase_train)
	images = np.array([np.float32(img) for img in images])	
	labels = np.array(labels)
	# labels = np.array([np.float32(label) for labels in images])
	kfold =KFold(n_splits=args.kfolds)
	
	for fold_index, (train_index, test_index) in enumerate(kfold.split(x_train)):
		print(f"fold_index:{fold_index}")

		x_train_fold = torch.tensor(x_train[train_index])		
		phase_train_fold = torch.tensor(phase_train[train_index])
		images_train_fold = torch.tensor(images[train_index])
		labels_train_fold = torch.tensor(labels[train_index])


		x_test_fold = torch.tensor(x_train[test_index])
		phase_test_fold = torch.tensor(phase_train[test_index])
		images_test_fold = torch.tensor(images[test_index])
		labels_test_fold = torch.tensor(labels[test_index])


		fold_train_dataset = TensorDataset(x_train_fold, phase_train_fold, images_train_fold, labels_train_fold)
		fold_val_dataset = TensorDataset(x_test_fold, phase_test_fold, images_test_fold, labels_test_fold)

		train_loader = DataLoader(dataset=fold_train_dataset,
								  batch_size=BATCH_SIZE, 
								  shuffle=True)

		valid_loader = DataLoader(dataset=fold_val_dataset, 
								  batch_size=BATCH_SIZE, 
								  shuffle=False)
		
		model, optimizer, _ = training_loop(model, criterion, optimizer, \
			train_loader, valid_loader, N_EPOCHS, DEVICE, fold_index=fold_index)

