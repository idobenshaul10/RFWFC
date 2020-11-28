import numpy as np
import cv2
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
import PIL
import torch
import torch.fft as fft

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

	def to_polar(self, x, y):
  		return (x**2 + y**2).sqrt(), torch.atan(y/x)
	
	def __getitem__(self, idx):		
		img, label = self.mnist_dataset[idx]		
		dft = fft.fftn(torch.tensor(np.float32(img)))
		mag, phase = self.to_polar(dft.real, dft.imag)
		# mag = transforms.Normalize(0, 1.)(mag.unsqueeze(0)).squeeze()
		return mag, dft, np.array(img), torch.tensor(label)

if __name__ == '__main__':	
	dataset = MnistPhaseDataset(train=True)
	mag, phase, gt_img, label = dataset[0]
	print(mag.shape)


	exit()
	dataset = datasets.MNIST(root=r'C:\datasets\mnist_data', 
		    train=True,
		   	download=True)

	img = np.array(dataset[0][0])
	dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
	mag, phase = cv2.cartToPolar(dft[:,:,0], dft[:,:,1])	
	real, imag = cv2.polarToCart(mag, phase)

	back = cv2.merge([real, imag])	
	img_back = cv2.idft(back)
	img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])	

	ax1 = plt.subplot(1,2,1)
	ax1.imshow(img, cmap='gray')

	ax2 = plt.subplot(1,2,2)
	ax2.imshow(img_back, cmap='gray')
	plt.show()