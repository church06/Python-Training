import os
import torch
from torch import nn
import pandas as pd
import itertools


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


def split_data(data_input: dict):
    data_split_fn = {}

    for i_fn in range(1, 469):
        sbj_fn = data_input[i_fn]
        key_list_fn = sbj_fn['sensor position'].unique().tolist()
        sbj_split_fn = {}

        for key_fn in key_list_fn:
            sbj_split_fn[key_fn] = sbj_fn[sbj_fn['sensor position'] == key_fn]

        data_split_fn[i_fn] = sbj_split_fn

    return data_split_fn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()


device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = get_data()
