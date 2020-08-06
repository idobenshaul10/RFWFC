import torch
import torchvision
import torchvision.transforms as transforms
import os
import sys
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
		if i % 2000 == 1999:    # print every 2000 mini-batches
			print('[%d, %5d] loss: %.3f' %
				  (epoch + 1, i + 1, running_loss / 2000))
			running_loss = 0.0

		if epoch % save_every == 0:			
			checkpoint_path = f"{output_path}/weights.{epoch}.h5"
			model_state_dict = model.state_dict()
			state_dict = OrderedDict()
			state_dict["epoch"] = epoch
			state_dict["checkpoint"] = model_state_dict						
			save(state_dict, checkpoint_path)

print('Finished Training')