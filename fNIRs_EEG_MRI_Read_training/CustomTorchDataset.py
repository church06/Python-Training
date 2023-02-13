import torch
import os
import pandas
from torch.utils.data import Dataset
from torchvision.io import read_file

path = 'E:/Coding/datasets/EEG/Kaggle_EEG_Alcohol/SMNI_CMI_TRAIN/Train/'
files = os.listdir(path)


class CustomImageDataset(Dataset):
    def __init__(self, data_fn, chanl_fn, idtfct_fn):
        self.data = pandas.read_csv(data_fn)
        self.channels = chanl_fn
        self.idtfct = idtfct_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idtfct_fn):
        self.data = ''


