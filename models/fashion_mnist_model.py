from torch import nn, optim
import torch.nn.functional as F

class FashionMnistModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.l1 = nn.Linear(784, 256)
		self.l2 = nn.Linear(256, 128)
		self.l3 = nn.Linear(128,64)
		self.l4 = nn.Linear(64,10)
		
		self.dropout = nn.Dropout(p=0.2)
	def forward(self,x):
		x = x.view(x.shape[0],-1)
		x = self.dropout(F.relu(self.l1(x)))
		x = self.dropout(F.relu(self.l2(x)))
		x = self.dropout(F.relu(self.l3(x)))
		x = F.log_softmax(self.l4(x), dim=1)
		return x