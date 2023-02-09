import os

import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy
from torch.utils.data import DataLoader
import torch.utils.data


def trains_print():
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

    return data_dict_fn


def formative_data_to_train(input_data_fn: dict):
    data_output_fn = {}

    for index_sbj_fn in input_data_fn:
        chanls = input_data_fn[index_sbj_fn]['sensor position'].unique().tolist()
        for_x_fn = input_data_fn[index_sbj_fn]
        for_y_fn = input_data_fn[index_sbj_fn]
        x_tensor = torch.empty()
        y_tensor = torch.empty()

        for chanl in chanls:
            x_fn = for_x_fn[for_x_fn['sensor position'] == chanl]['sensor value', 'time']
            y_fn = for_y_fn[for_y_fn['sensor position'] == chanl]['subject identifier']

    return data_output_fn


def show_data_fig(data_input_fn):
    for i in range(1, 3):
        plt.figure()
        plt.suptitle('FP1')
        for channel_fn in data_input_fn[i]['sensor position'].unique().tolist():
            _data_channel_fn = data_input_fn[i][data_input_fn[i]['sensor position'] == channel_fn]
            data_show_fn = _data_channel_fn['sensor value'].tolist()
            time_show_fn = data_input_fn[i]['FP1']['time'].tolist()

            plt.subplot(3, 4, i)
            plt.plot(time_show_fn, data_show_fn)
            plt.title(f"Subject: {'%03d' % i}", end='\r')
            plt.grid(True)

        plt.show()


def train_model(input_data_fn, model_fn, input_loss_fn, input_optimizer_fn, device_fn):
    for sbj_fn in input_data_fn:
        model_fn.train()

        x_fn = input_data_fn[sbj_fn][['sensor value', 'time']].to_numpy()
        y_fn = input_data_fn[sbj_fn]['subject identifier'].to_numpy()
        y_fn[y_fn == 'a'] = 1
        y_fn[y_fn == 'c'] = 0

        sbj_dataset_fn = torch.utils.data.TensorDataset(x_fn, y_fn)
        sbj_dataloader_fn = DataLoader(dataset=sbj_dataset_fn,
                                       batch_size=256,
                                       shuffle=True)

        size_fn = len(x_fn)

        for batch_fn, (x_to_model_fn, y_to_model_fn) in enumerate(sbj_dataloader_fn):
            x_to_model_fn, y_to_model_fn = x_to_model_fn.to(device_fn), y_to_model_fn.to(device_fn)

            pred_fn = model_fn(x_to_model_fn)
            loss_fn = input_loss_fn(pred_fn, y_fn)

            input_optimizer_fn.zero_grad()
            loss_fn.bacjward()
            input_optimizer_fn.step()

            if batch_fn % 100 == 0:
                loss_fn, current_fn = loss_fn.item(), batch_fn * len(x_fn)
                print(f'Loss: {loss_fn: >7d} [{current_fn: 5d}/{size_fn: >5d}]')


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten_fn = nn.Flatten()
        self.model_fn = nn.Sequential(
            nn.Linear(in_features=3 * 256,
                      out_features=3 * 256),

            nn.Linear(in_features=3 * 256,
                      out_features=int((3 * 256) / 2)),

            nn.Linear(in_features=int((3 * 256) / 2),
                      out_features=int((3 * 256) / 4)),

            nn.Linear(in_features=int((3 * 256) / 4),
                      out_features=2)
        )

    def forward(self, in_fn):
        x_fn = self.flatten_fn(in_fn)
        logits_fn = self.model_fn(x_fn)
        return logits_fn


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Use device: {device}")

data = get_data()
data_formative = formative_data_to_train(data)

model = Model().to(device)
print(model)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
loss = nn.CrossEntropyLoss()

epochs = 15
for epoch in range(epochs):
    print(f"{''.ljust(20, '-')}\nEpoch {epoch + 1}")
    train_model(data, model, loss, optimizer, device)
print('Done')
