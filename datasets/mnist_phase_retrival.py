import numpy as np
import cv2
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
import PIL
import torch

class MnistPhaseDataset(torch.utils.data.Dataset):
	def __init__(self, train=True, use_aug=True):		
		self.train = train
		if self.train:
			self.mnist_dataset = datasets.MNIST(root=r'C:\datasets\mnist_data', 
			    train=True,
			   	download=True)
		else:			
			self.mnist_dataset = datasets.MNIST(root=r'C:\datasets\mnist_data', 
			    train=False,
			   	download=True)

	def __len__(self):
		return len(self.mnist_dataset)
	
	def __getitem__(self, idx):
		img, __ = self.mnist_dataset[idx]
		dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
		mag, phase = cv2.cartToPolar(dft[:,:,0], dft[:,:,1])		
		return torch.tensor(mag), torch.tensor(dft), np.array(img)

dataset = datasets.MNIST(root=r'C:\datasets\mnist_data', 
	    train=False,
	   	download=True)


if __name__ == '__main__':
	img = np.array(dataset[0][0])
	dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
	mag, phase = cv2.cartToPolar(dft[:,:,0], dft[:,:,1])
	real, imag = cv2.polarToCart(mag, phase)

	back = cv2.merge([real, imag])
	import pdb; pdb.set_trace()
	img_back = cv2.idft(back)
	img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

	ax1 = plt.subplot(1,2,1)
	ax1.imshow(img, cmap='gray')

	ax2 = plt.subplot(1,2,2)
	ax2.imshow(img_back, cmap='gray')
	plt.show()