import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt

#Loading the MNIST dataset
def loadDataset():
    workers = 0
    batch_size = 20
    valid_size = 0.2

    transform = transforms.ToTensor()

    train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

    #Creating the validation Dataset
    num_train = len(train_data)
    indices = list(range(num_train))

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

    print("Train Loader", train_loader)

    return train_loader, valid_loader, test_loader

# def visualizeDataset(trainData):
#     #Obtaining One batch of training images
#     dataIter = iter(trainData)
#     print("Data Iter", dataIter)
#     images, labels = dataIter.next()
#     images = images.numpy()

#     #Visualizing the dataset
#     fig = plt.figure(figuresize=(25,4))
#     for idx in np.arange(20):
#         ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
#         ax.imshow(np.squeeze(images[idx]), cmap='grey')
#         ax.set_title(str(labels[idx].items()))

