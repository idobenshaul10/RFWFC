import torch
import torchvision
from torchvision import datasets,transforms
from torch import save
import os 
import inspect
import sys
from torch import nn, optim
from collections import OrderedDict
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from models.fashion_mnist_model import FashionMnistModel
import argparse

parser = argparse.ArgumentParser(description='train simple net on fahsion-mnist dataset')
parser.add_argument('--output_path', default=r"C:\projects\RFWFC\results\DL_layers\trained_models", 
	help='output_path for checkpoints')
args, _ = parser.parse_known_args()

output_path = os.path.join(args.output_path, "fahsion_mnist_model")
if not os.path.isdir(output_path):
	os.mkdir(output_path)


transform = transforms.Compose([transforms.ToTensor(),
							   transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

trainset = torchvision.datasets.FashionMNIST(
	root = r'C:\datasets\fashion_mnist',
	train = True,
	download = True,
	transform = transforms.Compose([
		transforms.ToTensor()                                 
	])
)
testset = torchvision.datasets.FashionMNIST(
	root = r'C:\datasets\fashion_mnist',
	train = False,
	download = True,
	transform = transforms.Compose([
		transforms.ToTensor()                                 
	])
)

trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=64)
testloader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=64)

model = FashionMnistModel()
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(),lr=0.005)

epochs = 50
save_every = 1

train_losses, test_losses = [],[]
for e in range(epochs):
	train_loss = 0
	test_loss = 0
	accuracy = 0
	for images, labels in trainloader:
		optimizer.zero_grad()
		op = model(images)
		loss = criterion(op, labels)
		train_loss += loss.item()
		loss.backward()
		optimizer.step()
	else:
		with torch.no_grad():
			model.eval()
			for images,labels in testloader:
				log_ps = model(images)
				prob = torch.exp(log_ps)
				top_probs, top_classes = prob.topk(1, dim=1)
				equals = labels == top_classes.view(labels.shape)
				accuracy += equals.type(torch.FloatTensor).mean()
				test_loss += criterion(log_ps, labels)
		model.train()
	print("Epoch: {}/{}.. ".format(e+1, epochs),
			  "Training Loss: {:.3f}.. ".format(train_loss/len(trainloader)),
			  "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
			  "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

	if e % save_every == 0:
		checkpoint_path = f"{output_path}/weights.{e}.h5"
		model_state_dict = model.state_dict()
		state_dict = OrderedDict()
		state_dict["epoch"] = e
		state_dict["checkpoint"] = model_state_dict		
		state_dict["valid_acc"] = accuracy/len(testloader)
		save(state_dict, checkpoint_path)

	train_losses.append(train_loss/len(trainloader))
	test_losses.append(test_loss/len(testloader))
