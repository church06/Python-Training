import os.path
from itertools import product

import bdpy
import h5py
import keras
import numpy
import numpy as np


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

    for sbj, roi, feature in product(subject, rois, layers):
        print('Subject:    %s' % sbj)
        print('ROI:        %s' % roi)
        print('Num voxels: %d' % voxel[roi])
        print('Feature:    %s' % feature)
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

        # select image for y
        y = img_feature.select(feature)
        y_label = img_feature.select('ImageID')

        # sort through the y in y_label of labels
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

        predict_y, real_y = algorithm_predict_feature(x_train, y_train, x_test, y_test, num_voxel=500)


def algorithm_predict_feature(x_train, y_train, x_test, y_test, num_voxel):
    n_unit = y_train.shape(1)

    print('Data normalizing...')

    nom_mean_x = np.mean(x_train, axis=0)
    nom_scale_x = np.std(x_train, axis=0, ddof=1)

    x_train = (x_train - nom_mean_x) / nom_scale_x
    x_test = (x_test - nom_mean_x) / nom_scale_x

    print('--------------------- Start predicting')

    y_true = numpy.array([])
    y_predict = numpy.array([])

    print('Loop start...')

    for i in range(n_unit):
        print('Loop %03d' % (i + 1))


# =========================================================================

# run Project
# ========================
main()
