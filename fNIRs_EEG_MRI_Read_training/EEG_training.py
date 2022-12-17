import mne
import tensorflow as tf
import pandas as pd
import os

path = 'E:/Entrance/Coding/datasets/EEG/Kaggle_EEG_Alcohol/SMNI_CMI_TRAIN/Train'

file_name = os.listdir(path)
file_path = path + '/' + file_name[0]
print(file_path)

data = pd.read_csv(file_path)
print(data)
