import mne
import tensorflow as tf
import pandas as pd
import os

path = 'E:/Coding/datasets/EEG/Kaggle_EEG_Alcohol/SMNI_CMI_TRAIN/Train/'
file_name = os.listdir(path)

data = {}
for name in file_name:
    data[name] = pd.read_csv(path + name)


