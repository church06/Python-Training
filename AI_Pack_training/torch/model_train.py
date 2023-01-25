import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os

os.environ['TORCH_HOME'] = 'E:/Coding/AI_models/PyTorch'

data_train = datasets.FashionMNIST(
    root='E:/Coding/datasets/torch/FashionMNIST',
    train=True,
    download=True,
    transform=ToTensor()
)

data_tset = datasets.FashionMNIST(
    root='E:/Coding/datasets/torch/FashionMNIST',
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = {0: 'T-shirt', 1: 'Trouser',
              2: 'Pullover', 3: 'Dress',
              4: 'Coat', 5: 'Sandal',
              6: 'Shirt', 7: 'Sneaker',
              8: 'Bag', 9: 'Ankle Boot'
              }

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3

for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(data_train), size=(1,)).item()
    img, label = data_train[sample_idx]

    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.title()
