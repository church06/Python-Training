import os.path
from itertools import product

import bdpy
import h5py
import keras
import numpy
import numpy as np
import tensorflow.keras.layers

from bdpy.ml import add_bias
from bdpy.preproc import select_top
from bdpy.stats import corrcoef


def main():
    subjects = {'s1': os.path.abspath('bxz056/data/Subject1.h5'),
                's2': os.path.abspath('bxz056/data/Subject2.h5'),
                's3': os.path.abspath('bxz056/data/Subject3.h5'),
                's4': os.path.abspath('bxz056/data/Subject4.h5'),
                's5': os.path.abspath('bxz056/data/Subject5.h5'),
                'imageFeature': os.path.abspath('bxz056/data/ImageFeatures.h5')}

    regine_of_interest = {'VC': 'ROI_VC = 1',
                          'LVC': 'ROI_LVC = 1',
                          'HVC': 'ROI_HVC = 1',
                          'V1': 'ROI_V1 = 1',
                          'V2': 'ROI_V2 = 1',
                          'V3': 'ROI_V3 = 1',
                          'V4': 'ROI_V4 = 1',
                          'LOC': 'ROI_LOC = 1',
                          'FFA': 'ROI_FFA = 1',
                          'PPA': 'ROI_PPA = 1'}

    voxel = {'VC': 1000,
             'LVC': 1000,
             'HVC': 1000,
             'V1': 500,
             'V2': 500,
             'V3': 500,
             'V4': 500,
             'LOC': 500,
             'FFA': 500,
             'PPA': 500}

    layers = ['cnn1', 'cnn2', 'cnn3', 'cnn4', 'cnn5', 'cnn6', 'cnn7', 'cnn8',
              'hmax1', 'hmax2', 'hmax3',
              'gist', 'sift']

    print('=======================================')
    print('Data loading...')

    dataset = {}
    image_feature = {}

    for person in subjects:

        file = h5py.File(subjects[person], 'r')

        if person != 'imageFeature':
            # Subject 1 ~ 5

            print(person, '---------------------')
            print('data: ', file.keys())

            dataset[person] = bdpy.BData(subjects[person])

        else:
            image_feature = bdpy.BData(subjects[person])

    # dataset & metadata collected

    print('\n=======================================')
    print('Analyzing...\n')

    data_prepare(subject=dataset, rois=regine_of_interest, img_feature=image_feature,
                 voxel=voxel, layers=layers)


def data_prepare(subject, rois, img_feature, voxel, layers):
    print('Start learning:')
    print('-----------------')

    for sbj, roi, layer in product(subject, rois, layers):
        print('Subject:    %s' % sbj)
        print('ROI:        %s' % roi)
        print('Num voxels: %d' % voxel[roi])
        print('Layer:    %s' % layer)
        print('===============================')

        # ---------------------------------------------
        # Subject {'s1': [...],
        #          's2': [...],
        #               ...
        #          's5': [...]}

        data = subject[sbj]  # data = 'sbj': [...]
        # ---------------------------------------------

        # rois: {'VC': 'ROI_VC = 1',
        #        'LVC': 'ROI_LVC = 1',
        #           ...
        #        'PPA': 'ROI_PPA = 1'}

        x = data.select(rois[roi])  # x = 'ROI_VC = 1' if roi = 'VC

        # get data type in subject fMRI
        data_type = data.select('DataType')  # Mark the training data, seen data and imagine data
        labels = data.select('stimulus_id')  # Use Stimulus ID as the order to sort images

        y = img_feature.select(layer)  # select the image feature which be marked layers[layer]
        y_label = img_feature.select('ImageID')  # get image id

        # sort through the y in y_label of labels, correspond with brain data
        y_sort = bdpy.get_refdata(y, y_label, labels)  # labels -> y_label -> y

        # Flatten(): transfer the shape from vertical to horizontal
        label_train = (data_type == 1).flatten()  # mark of training data
        label_test_seen = (data_type == 2).flatten()  # Index for subject see an image
        label_test_img = (data_type == 3).flatten()  # Index for subject imagine an image

        # test data, overlap seen and imagined value
        test = label_test_img + label_test_seen

        # get training & test data in x
        x_train = x[label_train, :]
        x_test = x[test, :]

        # get training & test data in y
        y_train = y_sort[label_train, :]
        y_test = y_sort[test, :]

        print('Predicting...')

        predict_y, real_y = algorithm_predict_feature(x_train=x_train, y_train=y_train,
                                                      x_test=x_test, y_test=y_test,
                                                      num_voxel=500)


def algorithm_predict_feature(x_train, y_train, x_test, y_test, num_voxel):
    n_unit = y_train.shape[1]

    print('Data normalizing...')

    # compute average of each column and return a (1, n) matrix
    nom_mean_x = np.mean(x_train, axis=0)

    # compute standard deviation of each column in x and return a (1, n) matrix
    nom_scale_x = np.std(x_train, axis=0, ddof=1)

    # normalize x
    x_train = (x_train - nom_mean_x) / nom_scale_x
    x_test = (x_test - nom_mean_x) / nom_scale_x

    print('--------------------- Start predicting')

    y_true = numpy.array([])
    y_predict = numpy.array([])

    print('Loop start...')

    for i in range(n_unit):
        print('Loop %03d' % (i + 1))

        # Get unit
        y_train_unit = y_train[:, i]
        y_test_unit = y_test[:, i]

        norm_mean_y = np.mean(y_train_unit, axis=0)
        std_y = np.std(y_train_unit, axis=0, ddof=1)

        norm_scale_y = None

        if std_y == 0:
            norm_scale_y = 1
        else:
            norm_scale_y = std_y

        y_train_unit = (y_train_unit - norm_mean_y) / norm_scale_y

        # select the voxel in column
        correlation = corrcoef(y_train_unit, x_train, var='col')

        x_train_unit, voxel_index = select_top(x_train, np.abs(correlation), num_voxel, axis=1, verbose=False)
        x_test_unit = x_test[:, voxel_index]

        # Add bias terms
        x_train_unit = add_bias(x_train_unit, axis=1)
        x_test_unit = add_bias(x_test_unit, axis=1)

        model = tensorflow.keras.Sequential(
            tensorflow.keras.layer.Conv2D()
        )

# =========================================================================

# run Project
# ========================
main()
