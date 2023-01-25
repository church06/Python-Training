import os
import torch
from torch import nn
import pandas as pd


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()


path = 'E:/Coding/datasets/EEG/Kaggle_EEG_Alcohol/SMNI_CMI_TRAIN/Train/'
file_name = os.listdir(path)

data = {}
for name in file_name:

    name_num = ''
    for i in name:
        if i in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            name_num += i

    data[int(name_num)] = pd.read_csv(path + name)

for i in range(468):
    num = i + 1
    sb_id = data[num]['subject identifier'].unique()
    if 'c' in sb_id:
        print(num)
        break

model = ''
