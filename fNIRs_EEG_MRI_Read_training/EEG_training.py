import os
import torch
from matplotlib import gridspec
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy


def get_data():
    path_fn = 'E:/Coding/datasets/EEG/Kaggle_EEG_Alcohol/SMNI_CMI_TRAIN/Train/'
    file_name_fn = os.listdir(path_fn)

    data_dict_fn = {}

    for name_fn in file_name_fn:

        name_num_fn = ''
        for i_fn in name_fn:
            if i_fn in ['0', '1', '2', '3', '4',
                        '5', '6', '7', '8', '9']:
                name_num_fn += i_fn

        name_int_fn = int(name_num_fn)

        data_fn = pd.read_csv(path_fn + name_fn)
        key_list_fn = data_fn['sensor position'].unique().tolist()
        sbj_split_fn = {}

        for key_fn in key_list_fn:
            sbj_split_fn[key_fn] = data_fn[data_fn['sensor position'] == key_fn]

        data_dict_fn[name_int_fn] = sbj_split_fn

        print(f"Subject No.{'%03d' % name_int_fn} State: {data_fn['subject identifier'].unique()}")

    return data_dict_fn


def split_data(data_input_fn: dict):
    data_split_fn = {}

    for i_fn in range(1, 469):
        sbj_fn = data_input_fn[i_fn]
        key_list_fn = sbj_fn['sensor position'].unique().tolist()
        sbj_split_fn = {}

        for key_fn in key_list_fn:
            sbj_split_fn[key_fn] = sbj_fn[sbj_fn['sensor position'] == key_fn]

        data_split_fn[i_fn] = sbj_split_fn

    return data_split_fn


def show_data_fig(data_input_fn):
    plt.figure()
    plt.suptitle('FP1')
    for i in range(1, 13):
        data_show_fn = data_input_fn[i]['FP1']['sensor value'].tolist()
        time_show_fn = data_input_fn[i]['FP1']['time'].tolist()

        plt.subplot(3, 4, i)
        plt.plot(time_show_fn, data_show_fn)
        plt.title(f"Subject: {'%03d' % i}", end='')
        plt.grid(True)

    plt.show()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten_fn = nn.Flatten()
        self.model_fn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=3*256, out_features=3*256),
            nn.ReLU(),
            nn.Linear(in_features=3*256, out_features=int((3*256)/2)),
            nn.ReLU(),
            nn.Linear(in_features=int((3*256)/2), out_features=int((3*256)/4)),
            nn.ReLU(),
            nn.Linear(in_features=int((3*256)/4), out_features=2)
        )

    def forward(self, in_fn):
        x_fn = self.flatten_fn(in_fn)
        logits_fn = self.model_fn(x_fn)
        return logits_fn


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Use device: {device}")

data = get_data()

model = Model().to(device)
print(model)

x = torch.rand(1, 2, 256, device=device)
logits = model(x)
pred_prob = nn.Softmax(dim=1)(logits)
y_pred = pred_prob.argmax(1)
print(f'Predict Class: {y_pred}')

print('Function of Flatten layer'.ljust(20, '='))
input_channel = torch.rand(3, 2, 256)
print(input_channel.size())

flatten = nn.Flatten()
flatten_channel = flatten(input_channel)
print(flatten_channel.size())

layer_1 = nn.Linear(in_features=2*256, out_features=2)
hidden_1 = layer_1(flatten_channel)
print(hidden_1.size())
print(''.ljust(20, '='))

print(f'Before ReLU: {hidden_1}\n\n')
hidden_1_relu = nn.ReLU()(hidden_1)
print(f'After ReLU: {hidden_1_relu}')

for sbj in data:
    for channel in data[sbj]:
        data_channel = data[sbj][channel][['sensor value', 'subject identifier', 'time']].to_numpy()


