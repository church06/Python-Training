import os
import os.path
from itertools import product

import slir
import bdpy
import h5py
import keras.layers
import numpy as np
import pandas as pd
import tensorflow.keras.optimizers
from bdpy import get_refdata
from bdpy.ml import add_bias
from bdpy.preproc import select_top
from bdpy.stats import corrcoef
from keras import Sequential
from matplotlib import pyplot as plt


# TODO: read article and do parameter optimization + model design

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

    file = None

    for person in subjects:

        file = h5py.File(subjects[person], 'r')

        if person != 'imageFeature':
            # Subject 1 ~ 5
            print(person, '---------------------')
            print('data: ', file.keys())

            dataset[person] = bdpy.BData(subjects[person])

        else:
            image_feature = bdpy.BData(subjects[person])

    file.close()

    print('--------------------- data ')
    print('s1: %s\n'
          's2: %s\n'
          's3: %s\n'
          's4: %s\n'
          's5: %s' % (dataset['s1'].dataset.shape,
                      dataset['s2'].dataset.shape,
                      dataset['s3'].dataset.shape,
                      dataset['s4'].dataset.shape,
                      dataset['s5'].dataset.shape))

    # dataset & metadata collected
    data_prepare(dataset, regine_of_interest, image_feature, layers, voxel)


def data_prepare(subject, rois, img_feature, layers, voxel):
    print('Data prepare:')
    print('-----------------')

    for sbj, roi, layer in product(subject, rois, layers):
        data = subject[sbj]
        x = data.select(rois[roi])
        print('-----------------------------------')
        print('roi: %s, subject: %s, layer: %s' % (roi, sbj, layer))

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

        pred_y, true_y = algorithm_predict_feature(x_train=x_train, y_train=y_train,
                                                   x_test=x_test, y_test=y_test,
                                                   num_voxel=voxel[roi], information=[sbj, roi, layer])

        # Separate results for perception and imagery tests
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

        print('Predict average seen feature: ', pred_y_pt_av)
        print('True average seen feature: ', true_y_pt_av)
        print('Predict average imaginary feature: ', pred_y_im_av)
        print('True average imaginary feature: ', true_y_im_av)

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

        results.to_csv(('bxz056/result_%s_%s' % (subject, roi, layer)))


def algorithm_predict_feature(x_train, y_train, x_test, y_test, num_voxel, information: list):

    n_iter = 50

    print('Learning started:')
    print('---------------------------------')
    # Normalize brian data (x)
    norm_mean_x = np.mean(x_train, axis=0)
    norm_scale_x = np.std(x_train, axis=0, ddof=1)

    x_train = (x_train - norm_mean_x) / norm_scale_x
    x_test = (x_test - norm_mean_x) / norm_scale_x

    # save predict value
    model = slir.SparseLinearRegression(n_iter=n_iter, prune_mode=1)
    y_pred_all = []

    for i in range(1000):

        # SIFT descriptor normalization
        y_train_unit = y_train[:, i]
        y_test_unit = y_test[:, i]

        norm_mean_y = np.mean(y_train_unit, axis=0)
        std_y = np.std(y_train_unit, axis=0, ddof=1)
        norm_scale_y = 1 if std_y == 0 else std_y

        y_train_unit = (y_train_unit - norm_mean_y) / norm_scale_y

        # correlate with y and x
        correlation = corrcoef(y_train_unit, x_train, var='col')

        x_train, voxel_index = select_top(x_train, np.abs(correlation),
                                          num_voxel, axis=1,
                                          verbose=False)

        x_test = x_test[:, voxel_index]

        # Add bias terms
        x_train = add_bias(x_train, axis=1)
        x_test = add_bias(x_test, axis=1)

        # ===================================================================================
        # define the neural network architecture (convolutional net) ------------------------

        print('Subject: %s, Roi: %s, Layer: %s, Voxel: %d' %
              (information[0], information[1], information[2], i))

        # Training and test
        try:
            model.fit(x_train, y_train_unit)  # Training
            y_pred = model.predict(x_test)  # Test

        except Exception as e:
            print('x_test_unit: ', x_test.shape)
            print('y_test_unit: ', y_test_unit.shape)
            print('x_train_unit: ', x_train.shape)
            print('y_train_unit: ', y_train_unit.shape)

            print(e)
            y_pred = np.zeros(y_test_unit.shape)
        # -----------------------------------------------------------------------------------
        # ===================================================================================

        y_pred = y_pred * norm_scale_y + norm_mean_y  # denormalize
        print('loss: ', y_test[:, i] - y_pred)

        y_pred_all.append(y_pred)

    y_predicted = np.vstack(y_pred_all).T

    return y_predicted, y_test


def LinearModel(optimizer: ('adam', 'sdg'), learning_rate: float, loss_type: str):
    model = Sequential()

    model.add(keras.layers.Dense(units=1000, input_dim=1))
    model.add(keras.layers.Activation('linear'))
    model.add(keras.layers.Dense(units=1))

    if optimizer == 'adam':
        opr = tensorflow.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(opr, loss_type)

    elif optimizer == 'sdg':
        opr = tensorflow.keras.optimizers.SGD(learning_rate)
        model.compile(opr, loss_type)

    return model


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
