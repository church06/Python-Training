import pandas
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Dataset Fix Path
data_type = 'G:/Entrance/Coding/datasets/EEG'

# Dataset Name
dataset_name = 'Kaggle_EEG_Alcohol'

# Dataset Type
training_set = 'SMNI_CMI_TRAIN'
testing_set = 'SMNI_CMI_TEST'

# Notes
dataset_path = data_type + '/' + dataset_name
print('Read Dataset: ' + dataset_path)

print('Training Dataset'.ljust(18, ' ') + '|  ' + training_set.ljust(16, ' ')
      + '|  Path: ' + dataset_path + '/' + training_set)

print('Testing Dataset'.ljust(18, ' ') + '|  ' + testing_set.ljust(16, ' ')
      + '|  Path: ' + dataset_path + '/' + testing_set)

print('------------------------------------------')

# Main ====================================
trainSet_path = dataset_path + '/' + training_set
testSet_path = dataset_path + '/' + testing_set

train_A_data = {}
train_C_data = {}

test_A_data = {}
test_C_data = {}

for i in range(0, 480):
    data_mark = i + 1

    if data_mark <= 468:
        train_pd = pandas.read_csv(trainSet_path + '/Data{0}.csv'.format(data_mark))
        test_pd = pandas.read_csv(testSet_path + '/Data{0}.csv'.format(data_mark))

        train_identifier = train_pd['subject identifier'].unique()
        test_identifier = test_pd['subject identifier'].unique()

        if 'a' in train_identifier and 'c' not in train_identifier:
            train_A_data[data_mark] = train_pd
            test_A_data[data_mark] = test_pd
        else:
            train_C_data[data_mark] = train_pd
            test_C_data[data_mark] = test_pd

        if data_mark == 468:
            print('Get [Train, Test] data: [{0}/468, {0}/480]'.format(data_mark))
            print('Train data Collected.')
            print('------------------------------------------')
        else:
            print('Get [Train, Test] data: [{0}/468, {0}/480]'.format(data_mark), end='\r')

    else:
        test_pd = pandas.read_csv(testSet_path + '/Data{0}.csv'.format(data_mark))
        test_identifier = test_pd['subject identifier'].unique()

        if 'a' in test_identifier and 'c' not in test_identifier:
            test_A_data[data_mark] = test_pd
        else:
            test_C_data[data_mark] = test_pd

        if data_mark == 480:
            print('Get [Test] data: [{0}/480]'.format(data_mark))
            print('Test data Collected.')
            print('------------------------------------------')
        else:
            print('Get [Test] data: [{0}/480]'.format(data_mark), end='\r')

print('All data collected.')

model = tf.keras.Sequential([
    
])
