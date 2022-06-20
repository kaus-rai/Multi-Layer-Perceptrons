import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt

workers = 0
batch_size = 20
valid_size = 0.2

transform = transforms.ToTensor()

train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

#Creating the validation Dataset
num_train = len(train_data)
indices = list(range(num_train))

print("INdiicaes", indices)
np.random.shuffle(indices)
split = int(np.floor(valid_size*num_train))
train_idx, valid_idx = indices[split:], indices[:split]

#Getting the training and test data
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

#Preparing the data loader
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, sampler=valid_sampler, num_workers=workers)

#Obtaining One batch of training images
dataIter = iter(train_loader)
images, labels = dataIter.next()
images = images.numpy()

#Visualizing the dataset