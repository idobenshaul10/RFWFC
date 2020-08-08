import torch
import torchvision
import torchvision.transforms as transforms
import os
import sys
from datetime import datetime
import inspect
import torch.nn as nn
import torch.nn.functional as F
from torch import save
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from models.simple_cifar10_model import Simple_CIFAR10
import torch.optim as optim
import argparse
from collections import OrderedDict

parser = argparse.ArgumentParser(description='train conv-net on cifar10 dataset')
parser.add_argument('--output_path', default=r"C:\projects\RFWFC\results\DL_layers\trained_models", 
	help='output_path for checkpoints')
args, _ = parser.parse_known_args()
use_cuda = torch.cuda.is_available()

epochs = 100
save_every = 5
output_path = os.path.join(args.output_path, "cifar10_simple")
if not os.path.isdir(output_path):
	os.mkdir(output_path)

transform = transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root=r'C:\datasets\cifar10', train=True,
										download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
										  shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root=r'C:\datasets\cifar10', train=False,
									   download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
										 shuffle=False, num_workers=0)




model = Simple_CIFAR10()
if use_cuda:
	model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

print_every = 1

def get_accuracy(model, data_loader):	
	correct_pred = 0
	n = 0
	with torch.no_grad():
		model.eval()
		for X, y_true in data_loader:			
			y_true, X = y_true.cuda(), X.cuda()
			y_prob = model(X)			
			predicted_labels = torch.max(y_prob, 1)[1]
			n += y_true.size(0)			
			correct_pred += (predicted_labels == y_true).sum()
	return correct_pred.float() / n


for epoch in range(epochs):  # loop over the dataset multiple times
	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):
		inputs, labels = data
		if use_cuda:
			inputs, labels = inputs.cuda(), labels.cuda()
		optimizer.zero_grad()

		outputs = model(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		
	running_loss += loss.item()
	if epoch % print_every == (print_every - 1):			
		train_acc = get_accuracy(model, trainloader)
		valid_acc = get_accuracy(model, testloader)
			
		print(f'{datetime.now().time().replace(microsecond=0)} --- '
			  f'Epoch: {epoch}\t'
			  f'Train loss: {running_loss:.4f}\t'			  
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

print('Finished Training')