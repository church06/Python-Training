import os

import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy
from torch.utils.data import DataLoader
import torch.utils.data


def trainings_print():
    print('Function of Flatten layer'.ljust(20, '='))
    input_channel = torch.rand(3, 2, 256)
    print(input_channel.size())

    flatten = nn.Flatten()
    flatten_channel = flatten(input_channel)
    print(f'Flattened layer: {flatten_channel.size()}')

    layer_1_fn = nn.Linear(in_features=2 * 256, out_features=2)
    hidden_1_fn = layer_1_fn(flatten_channel)
    print(f'Hidden layer 1: {hidden_1_fn.size()}')
    print(f'Before ReLU: {hidden_1_fn}\n\n')

    hidden_1_relu_fn = nn.ReLU()(hidden_1_fn)
    print(f'After ReLU: {hidden_1_relu_fn}')
    print(''.ljust(20, '='))

    x_fn = torch.rand(1, 2, 256, device=device)
    logits_fn = model(x_fn)
    pred_prob_fn = nn.Softmax(dim=1)(logits_fn)
    y_pred_fn = pred_prob_fn.argmax(1)
    print(f'Predict Class: {y_pred_fn}')


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
        data_dict_fn[name_int_fn] = data_fn

        print(f"Subject No.{'%03d' % name_int_fn} State: {data_fn['subject identifier'].unique()}", end='\r')

    print()
    print('Done.')
    return data_dict_fn


def formative_data_to_train(data_in_fn: dict):
    data_formative_fn = {}

    for index_sbj_fn in data_in_fn:
        sbj_dict_fn = {}
        chanls = data_in_fn[index_sbj_fn]['sensor position'].unique().tolist()
        for_x_fn = data_in_fn[index_sbj_fn]
        for_y_fn = data_in_fn[index_sbj_fn]
        x_tensor = []
        y_tensor = []

        for chanl in chanls:
            x_fn = torch.tensor(for_x_fn[for_x_fn['sensor position'] == chanl][['sensor value', 'time']].to_numpy())
            y_fn = for_y_fn[for_y_fn['sensor position'] == chanl]['subject identifier'].to_numpy()

            y_fn[y_fn == 'a'] = 1
            y_fn[y_fn == 'c'] = 0
            y_fn = torch.tensor(y_fn.astype(numpy.int8))

            x_tensor.append(x_fn)
            y_tensor.append(y_fn)
            print(f"Subject: {index_sbj_fn:03d} | Channel: {chanl.ljust(3, ' ')} | "
                  f"X tensor: {len(x_tensor) :03d} | Y tensor: {len(y_tensor):03d}",
                  end='\r')
        print()

        sbj_dict_fn['x'] = torch.stack(x_tensor)
        sbj_dict_fn['y'] = torch.stack(y_tensor)
        data_formative_fn[index_sbj_fn] = sbj_dict_fn

    print()
    print('Done.')

    return data_formative_fn


def show_data_fig(data_in_fn):
    for i in range(1, 3):
        plt.figure()
        plt.suptitle('FP1')
        for chanl_fn in data_in_fn[i]['sensor position'].unique().tolist():
            data_chanl_fn = data_in_fn[i][data_in_fn[i]['sensor position'] == chanl_fn]
            data_show_fn = data_chanl_fn['sensor value'].tolist()
            time_show_fn = data_in_fn[i]['FP1']['time'].tolist()

            plt.subplot(3, 4, i)
            plt.plot(time_show_fn, data_show_fn)
            plt.title(f"Subject: {'%03d' % i}", end='\r')
            plt.grid(True)

        plt.show()


def train_model(data_in_fn, model_fn, input_loss_fn, input_optimizer_fn, device_fn):
    for sbj_fn in data_in_fn:
        model_fn.train()

        x_all_fn = data_in_fn[sbj_fn]['x']
        y_all_fn = data_in_fn[sbj_fn]['y']

        for chanl_fn in range(64):
            x_fn = x_all_fn[chanl_fn]
            y_gn = y_all_fn[chanl_fn]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten_fn = nn.Flatten()
        self.model_fn = nn.Sequential(
            nn.Linear(in_features=256 * 2, out_features=256 * 2),

            nn.Linear(in_features=256 * 2, out_features=int((256 * 2) / 2)),

            nn.Linear(in_features=int((256 * 2) / 2), out_features=int((256 * 2) / 4)),

            nn.Linear(in_features=int((256 * 2) / 4), out_features=2)
        )

    def forward(self, in_fn):
        x_fn = self.flatten_fn(in_fn)
        logits_fn = self.model_fn(x_fn)
        return logits_fn


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Use device: {device}")

data = formative_data_to_train(get_data())

model = Model().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
loss = nn.CrossEntropyLoss()
print(model)

epochs = 15
for epoch in range(epochs):
    train_model(data, model, loss, optimizer, device)
    print(f"{''.ljust(20, '-')}\nEpoch {epoch + 1}")
print('Done')
