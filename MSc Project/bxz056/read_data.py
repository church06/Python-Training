import os
import os.path
from itertools import product

import bdpy
import h5py
import numpy
import numpy as np
import tensorflow.keras as keras
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
    print('\n=======================================')
    print('Analyzing...\n')

    data_prepare(dataset, regine_of_interest, image_feature, layers, voxel)


def data_prepare(subject, rois, img_feature, layers, voxel):
    print('Start learning:')
    print('-----------------')

    for sbj, roi, layer in product(subject, rois, layers):
        data = subject[sbj]
        x = data.select(rois[roi])
        print('roi: %s, x: %s' % (roi, x.shape))

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

        algorithm_predict_feature(x_train=x_train, y_train=y_train,
                                  x_test=x_test, y_test=y_test,
                                  num_voxel=voxel[roi])


def algorithm_predict_feature(x_train, y_train, x_test, y_test, num_voxel):
    print('--------------------- Start predicting')

    # Normalize brian data (x)
    norm_mean_x = np.mean(x_train, axis=0)
    norm_scale_x = np.std(x_train, axis=0, ddof=1)

    x_train = (x_train - norm_mean_x) / norm_scale_x
    x_test = (x_test - norm_mean_x) / norm_scale_x

    print('Loop start...')

    # Normalize image features for training (y_train_unit)
    # norm_mean_y = np.mean(y_train, axis=0)
    # std_y = np.std(y_train, axis=0, ddof=1)
    #
    # norm_scale_y = numpy.array([])
    #
    # for i in std_y:
    #     if i == 0:
    #         norm_scale_y = numpy.append(norm_scale_y, 1)
    #     else:
    #         norm_scale_y = numpy.append(norm_scale_y, i)
    #
    # y_train = (y_train - norm_mean_y) / norm_scale_y

    # correlate with y and x
    correlation = corrcoef(y_train[:, 0], x_train, var='col')

    x_train, voxel_index = select_top(x_train, np.abs(correlation),
                                      num_voxel, axis=1,
                                      verbose=False)

    print('voxel_index: ', voxel_index.shape)
    x_test = x_test[:, voxel_index]

    # Add bias terms
    # x_train = add_bias(x_train, axis=1)
    # x_test = add_bias(x_test, axis=1)

    # Training dataset shape
    x_axis_0 = x_train.shape[0]
    x_axis_1 = x_train.shape[1]
    print('x_axis_0: %s, x_axis_1: %s' %(x_axis_0, x_axis_1))

    # Test dataset shape
    xt_axis_0 = x_test.shape[0]

    # print('x_train: ', x_train.shape)
    # print('x_test: ', x_test.shape)

    # Reshape for Conv2D
    if x_train.shape[1] == 1000:
        x_train = x_train.reshape(x_axis_0, 1000, 1, 1)
        x_test = x_test.reshape(xt_axis_0, 1000, 1, 1)

        y_train = y_train.reshape(x_axis_0, 40, 25, 1)
        y_test = y_test.reshape(xt_axis_0, 40, 25, 1)

    else:
        x_train = x_train.reshape(x_axis_0, 32, 32, 1)
        x_test = x_test.reshape(xt_axis_0, 32, 32, 1)

        y_train = y_train.reshape(x_axis_0, 32, 32, 1)
        y_test = y_test.reshape(xt_axis_0, 32, 32, 1)

    # print('after reshape x_train: ', x_train.shape)
    # print('after reshape x_test: ', x_test.shape)
    # print('y_train: ', y_train.shape)
    # print('y_test: ', y_test.shape)

    layer_axis_0 = x_train.shape[1]
    layer_axis_1 = x_train.shape[2]

    # define the neural network architecture (convolutional net)
    model = Sequential()

    model.add(keras.layers.Conv2D(filters=8,
                                  input_shape=(layer_axis_0, layer_axis_1, 1),
                                  kernel_size=3,
                                  activation='relu'
                                  ))
    model.add(keras.layers.AvgPool2D(padding='same'))

    model.add(keras.layers.Conv2D(filters=8,
                                  kernel_size=3,
                                  activation='relu'
                                  ))
    model.add(keras.layers.AvgPool2D(padding='same'))

    model.add(keras.layers.Dense(1000, activation='softmax'))

    optimizer = Adam()
    loss = keras.losses.mean_squared_error

    model.compile(optimizer, loss, metrics=['MeanSquaredError'])

    # Training and test
    print(model.summary())
    model.fit(x_train, y_train, epochs=200)  # Training
    y_pred_list = model.predict(x_test)  # Test

    # y_pred_list = y_pred_list * norm_scale_y + norm_mean_y

    print('y_pred_list: ', y_pred_list.shape)
    print(y_test.shape)

    y_pred_list = y_pred_list.reshape(y_test.shape[0], y_test.shape[1])
    print('y_pred_list after reshape: ', y_pred_list.shape)


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
