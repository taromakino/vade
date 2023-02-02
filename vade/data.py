import os
import torch
import torchvision.transforms as T
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST

def make_data(data_dpath, batch_size, n_workers):
    os.makedirs(data_dpath, exist_ok=True)
    transform = T.Compose([T.ToTensor(), T.Lambda(torch.flatten)])
    trainval_data = MNIST(data_dpath, train=True, download=True, transform=transform)
    test_data = MNIST(data_dpath, train=False, download=True, transform=transform)
    train_data, val_data = random_split(trainval_data, [55000, 5000])

    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=n_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=n_workers, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=n_workers, shuffle=False)
    return train_loader, val_loader, test_loader