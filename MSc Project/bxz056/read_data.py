import os
import os.path
from itertools import product

import bdpy
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow.keras as keras
from bdpy import get_refdata
from bdpy.ml import add_bias
from bdpy.preproc import select_top
from bdpy.stats import corrcoef
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


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

    data_prepare(dataset, regine_of_interest, image_feature, layers, voxel)


def data_prepare(subject, rois, img_feature, layers, voxel):
    print('Start learning:')
    print('-----------------')

    for sbj, roi, layer in product(subject, rois, layers):
        print('--------------------')
        print('Subject:    %s' % sbj)
        print('ROI:        %s' % roi)
        print('Num voxels: %d' % voxel[roi])
        print('Layers:    %s' % layer)

        data = subject[sbj]
        # ---------------------------------------------

        x = data.select(rois[roi])
        # --------------------------------------------

        data_type = data.select('DataType')
        labels = data.select('stimulus_id')

        y = img_feature.select(layer)
        y_label = img_feature.select('ImageID')

        y_sort = bdpy.get_refdata(y, y_label, labels)  # labels -> y_label -> y

        # Flatten(): transfer the shape from vertical to horizontal
        i_train = (data_type == 1).flatten()
        i_test_seen = (data_type == 2).flatten()
        i_test_img = (data_type == 3).flatten()

        # test data, add seen and imagined data together
        i_test = i_test_img + i_test_seen

        # get training & test data in x
        x_train = x[i_train, :]
        x_test = x[i_test, :]

        # get training & test data in y
        y_train = y_sort[i_train, :]
        y_test = y_sort[i_test, :]

        true_y, pred_y = algorithm_predict_feature(x_train=x_train, y_train=y_train,
                                                   x_test=x_test, y_test=y_test,
                                                   num_voxel=voxel[roi])

        i_pt = i_test_seen[i_test]  # Index for perception test within test
        i_im = i_test_img[i_test]  # Index for imagery test within test

        pred_y_pt = pred_y[i_pt, :]
        pred_y_im = pred_y[i_im, :]

        true_y_pt = true_y[i_pt, :]
        true_y_im = true_y[i_im, :]

        # Get averaged predicted feature
        test_label_pt = labels[i_test_seen, :].flatten()
        test_label_im = labels[i_test_img, :].flatten()

        pred_y_pt_av, true_y_pt_av, test_label_set_pt \
            = get_averaged_feature(pred_y_pt, true_y_pt, test_label_pt)
        pred_y_im_av, true_y_im_av, test_label_set_im \
            = get_averaged_feature(pred_y_im, true_y_im, test_label_im)

        # Get category averaged features
        catlabels_pt = np.vstack([int(n) for n in test_label_pt])  # Category labels (perception test)
        catlabels_im = np.vstack([int(n) for n in test_label_im])  # Category labels (imagery test)
        catlabels_set_pt = np.unique(catlabels_pt)  # Category label set (perception test)
        catlabels_set_im = np.unique(catlabels_im)  # Category label set (imagery test)

        y_catlabels = img_feature.select('CatID')  # Category labels in image features
        ind_catave = (img_feature.select('FeatureType') == 3).flatten()

        y_catave_pt = get_refdata(y[ind_catave, :], y_catlabels[ind_catave, :], catlabels_set_pt)
        y_catave_im = get_refdata(y[ind_catave, :], y_catlabels[ind_catave, :], catlabels_set_im)

        # Prepare result dataframe
        results = pd.DataFrame({'subject': [sbj, sbj],
                                'roi': [roi, roi],
                                'feature': [layer, layer],
                                'test_type': ['perception', 'imagery'],
                                'true_feature': [true_y_pt, true_y_im],
                                'predicted_feature': [pred_y_pt, pred_y_im],
                                'test_label': [test_label_pt, test_label_im],
                                'test_label_set': [test_label_set_pt, test_label_set_im],
                                'true_feature_averaged': [true_y_pt_av, true_y_im_av],
                                'predicted_feature_averaged': [pred_y_pt_av, pred_y_im_av],
                                'category_label_set': [catlabels_set_pt, catlabels_set_im],
                                'category_feature_averaged': [y_catave_pt, y_catave_im]})

        print(results)


def algorithm_predict_feature(x_train, y_train, x_test, y_test, num_voxel):
    print('--------------------- Start predicting')

    n_unit = y_train.shape[1]

    # Normalize brian data (x)
    norm_mean_x = np.mean(x_train, axis=0)
    norm_scale_x = np.std(x_train, axis=0, ddof=1)

    x_train = (x_train - norm_mean_x) / norm_scale_x
    x_test = (x_test - norm_mean_x) / norm_scale_x

    y_true_list = []
    y_pred_list = []

    print('Loop start...')

    for i in range(n_unit):
        # Get unit
        y_train_unit = y_train[:, i]
        y_test_unit = y_test[:, i]

        # Normalize image features for training (y_train_unit)
        norm_mean_y = np.mean(y_train_unit, axis=0)
        std_y = np.std(y_train_unit, axis=0, ddof=1)
        norm_scale_y = 1 if std_y == 0 else std_y

        y_train_unit = (y_train_unit - norm_mean_y) / norm_scale_y

        # correlate with y and x
        correlation = corrcoef(y_train_unit, x_train, var='col')

        x_train_unit, voxel_index = select_top(x_train, np.abs(correlation), num_voxel, axis=1, verbose=False)
        x_test_unit = x_test[:, voxel_index]

        # Add bias terms
        x_train_unit = add_bias(x_train_unit, axis=1)
        x_test_unit = add_bias(x_test_unit, axis=1)

        # Training dataset shape
        x_axis_0 = x_train_unit.shape[0]
        x_axis_1 = x_train_unit.shape[1]

        # Test dataset shape
        xt_axis_0 = x_test_unit.shape[0]
        xt_axis_1 = x_test_unit.shape[1]

        # Reshape for Conv1D
        x_train_unit = x_train_unit.reshape(x_axis_0, x_axis_1, 1, 1)
        x_test_unit = x_test_unit.reshape(xt_axis_0, xt_axis_1, 1, 1)

        # define the neural network architecture (convolutional net)
        model = Sequential()

        model.add(keras.layers.Conv1D(filters=1,
                                      kernel_size=1001,
                                      input_shape=(1001, 1),
                                      padding='valid',
                                      ))
        model.add(keras.layers.AveragePooling1D(1, padding='same'))

        optimizer = Adam(learning_rate=0.00000000000001)
        loss = keras.losses.categorical_crossentropy

        model.compile(optimizer, loss)

        # Training and test
        model.fit(x_train_unit, y_train_unit)  # Training
        y_pred = model.predict(x_test_unit)  # Test

        y_pred = y_pred * norm_scale_y + norm_mean_y

        y_true_list.append(y_test_unit)
        y_pred_list.append(y_pred)

    # Create numpy arrays for return values
    y_predicted = np.vstack(y_pred_list).T
    y_true = np.vstack(y_true_list).T

    return y_predicted, y_true


def get_averaged_feature(pred_y, true_y, labels):
    """Return category-averaged features"""

    labels_set = np.unique(labels)

    pred_y_av = np.array([np.mean(pred_y[labels == c, :], axis=0) for c in labels_set])
    true_y_av = np.array([np.mean(true_y[labels == c, :], axis=0) for c in labels_set])

    return pred_y_av, true_y_av, labels_set


# =========================================================================

# run Project
# ========================
main()
