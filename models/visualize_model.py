
import numpy as np
import torch
import torchvision.models
import hiddenlayer as hl
from fashion_mnist_bad_net import fashion_mnist_bad_net

model = fashion_mnist_bad_net()
hl.build_graph(model, torch.zeros([1, 3, 28, 28]))


