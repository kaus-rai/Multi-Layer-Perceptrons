import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from datasetHelper import loadDataset
import neural


def trainModel():
    train, val, test = loadDataset()

    neuralModel = neural.MlpNet()
    print(neuralModel)

trainModel()
