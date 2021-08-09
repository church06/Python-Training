import os
import os.path
import pickle
from itertools import product

import bdpy
import h5py
import numpy
import pandas as pd
import slir
from bdpy import get_refdata, makedir_ifnot
from bdpy.ml import add_bias
from bdpy.preproc import select_top
from bdpy.stats import corrcoef
from matplotlib import pyplot as plt


# TODO: Data preprocessing through different normalization method

def data_prepare(subject, rois, img_feature, layer_all, voxel_all, repeat):
    print('Data prepare:')
    print('-----------------')

    # For plot ========================================================

    none = []
    z = []
    min_max = []
    normal_ds = []

    # For plot ========================================================

    for sbj, roi, layer in product(subject, rois, layer_all):
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

        # for test -------------
        need_axis_0 = 2

        x_train = x_train[:need_axis_0, :]
        x_test = x_test[:need_axis_0, :]

        y_train = y_train[:need_axis_0, :]
        y_test = y_test[:need_axis_0, :]
        # ----------------------

        for i in range(repeat):

            if i == 0:
                print('No normalizing =============================')
                pred_y, true_y = algorithm_predict_feature(x_train=x_train, y_train=y_train,
                                                           x_test=x_test, y_test=y_test,
                                                           num_voxel=voxel_all[roi], information=[sbj, roi, layer],
                                                           norm=i)
                none = numpy.append(pred_y, true_y, axis=0)

            elif i == 1:
                print('Z-score normalization using ================')
                pred_y, true_y = algorithm_predict_feature(x_train=x_train, y_train=y_train,
                                                           x_test=x_test, y_test=y_test,
                                                           num_voxel=voxel_all[roi], information=[sbj, roi, layer],
                                                           norm=i)
                z = numpy.append(pred_y, true_y, axis=0)

            elif i == 2:
                print('Min-Max normalization using ================')
                pred_y, true_y = algorithm_predict_feature(x_train=x_train, y_train=y_train,
                                                           x_test=x_test, y_test=y_test,
                                                           num_voxel=voxel_all[roi], information=[sbj, roi, layer],
                                                           norm=i)
                min_max = numpy.append(pred_y, true_y, axis=0)

            elif i == 3:
                print('Binary normalization using =================')
                pred_y, true_y = algorithm_predict_feature(x_train=x_train, y_train=y_train,
                                                           x_test=x_test, y_test=y_test,
                                                           num_voxel=voxel_all[roi], information=[sbj, roi, layer],
                                                           norm=i)
                normal_ds = numpy.append(pred_y, true_y, axis=0)

        # ----------------------------------------------------- Return setting
        if repeat == 1:
            return none
        elif repeat == 2:
            return none, z
        elif repeat == 3:
            return none, z, min_max
        elif repeat == 4:
            return none, z, min_max, normal_ds

        # --------------------------------------------------------------------

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
        cat_labels_pt = numpy.vstack([int(n) for n in test_label_pt])  # Category labels (perception test)
        cat_labels_im = numpy.vstack([int(n) for n in test_label_im])  # Category labels (imagery test)
        cat_labels_set_pt = numpy.unique(cat_labels_pt)  # Category label set (perception test)
        cat_labels_set_im = numpy.unique(cat_labels_im)  # Category label set (imagery test)

        y_cat_labels = img_feature.select('CatID')  # Category labels in image features
        ind_cat_av = (img_feature.select('FeatureType') == 3).flatten()

        y_cat_av_pt = get_refdata(y[ind_cat_av, :], y_cat_labels[ind_cat_av, :], cat_labels_set_pt)
        y_cat_av_im = get_refdata(y[ind_cat_av, :], y_cat_labels[ind_cat_av, :], cat_labels_set_im)

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
                                'category_label_set': [cat_labels_set_pt, cat_labels_set_im],
                                'category_feature_averaged': [y_cat_av_pt, y_cat_av_im]})

        # Save results
        analysis_id = sbj + '-' + roi + '-' + layer
        results_file = os.path.join('bxz056/plots', analysis_id + '.pkl')

        makedir_ifnot(os.path.dirname(results_file))
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)

        print('Saved %s' % results_file)


def avg_loss(array):
    loss_avg = 0

    for j in array:
        loss_avg += j

    return loss_avg / 1000


def algorithm_predict_feature(x_train, y_train, x_test, y_test, num_voxel, information: list, norm: (0, 1, 2, 3)):
    # Plot setting for not python scientific model
    # plt.figure(figsize=(10, 8))

    # Training iteration
    n_iter = 200

    # Normalize brian data (x) --------------------------
    if norm == 0:
        x_train = x_train
        x_test = x_test

    elif norm == 1:
        # Z-Score
        norm_mean_x = numpy.mean(x_train, axis=0)
        norm_scale_x = numpy.std(x_train, axis=0, ddof=1)

        x_train = (x_train - norm_mean_x) / norm_scale_x
        x_test = (x_test - norm_mean_x) / norm_scale_x

    elif norm == 2:
        # Min-Max
        x_min = numpy.min(x_train)
        x_max = numpy.max(x_train)

        x_train = (x_train - x_min) / (x_max - x_min)
        x_test = (x_test - x_min) / (x_max - x_min)

    elif norm == 3:
        # Decimal Scaling
        x_train_abs = numpy.abs(x_train)
        x_abs_max = numpy.max(x_train_abs)

        power = 0

        while x_abs_max < 1:
            x_abs_max /= 10
            power += 1

        x_train = x_train / numpy.power(10, power)
        x_test = x_test / numpy.power(10, power)
    # ---------------------------------------------------

    # save predict value
    model = slir.SparseLinearRegression(n_iter=n_iter, prune_mode=1)
    y_pred_all = []

    # ===================================================================================
    # define the neural network architecture (convolutional net) ------------------------

    for i in range(1000):

        print('Subject: %s, Roi: %s, Layer: %s, Voxel: %d' %
              (information[0], information[1], information[2], i))

        # Image feature normalization - for all data
        y_train_unit = y_train[:, i]
        y_test_unit = y_test[:, i]
        # ------------------------------------------

        # Normalization of image feature -----------------------------
        # Z-Score -----------------------------------
        norm_mean_y = numpy.mean(y_train_unit, axis=0)
        std_y = numpy.std(y_train_unit, axis=0, ddof=1)
        norm_scale_y = 1 if std_y == 0 else std_y
        # ---------------------------------------------

        # Min-Max ----------------
        y_min = numpy.min(x_train)
        y_max = numpy.max(x_train)
        # ------------------------

        # Decimal Scaling ----------------
        y_train_abs = numpy.abs(y_train)
        y_abs_max = numpy.max(y_train_abs)

        power = 0
        while y_abs_max < 1:
            y_abs_max /= 10
            power += 1
        # --------------------------------

        if norm == 0:
            # No normalization
            y_train_unit = y_train_unit
            y_test_unit = y_test_unit

        elif norm == 1:
            # z-score
            y_train_unit = (y_train_unit - norm_mean_y) / norm_scale_y

        elif norm == 2:
            # min-max
            y_train = (y_train - y_min) / (y_max - y_min)

        elif norm == 3:
            # Decimal Scaling
            y_train = y_train / numpy.power(10, power)
        # ------------------------------------------------------------

        # correlate with y and x------------------------------------------
        correlation = corrcoef(y_train_unit, x_train, var='col')

        x_train, voxel_index = select_top(x_train, numpy.abs(correlation),
                                          num_voxel, axis=1,
                                          verbose=False)

        # ----------------------------------------------------------------

        # Get x test --------------------
        x_test = x_test[:, voxel_index]
        # -------------------------------

        # Add bias terms ------------------
        x_train = add_bias(x_train, axis=1)
        x_test = add_bias(x_test, axis=1)
        # ---------------------------------

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
            y_pred = numpy.zeros(y_test_unit.shape)

        # Denormalize ----------------------------------
        if norm == 1:
            y_pred = y_pred * norm_scale_y + norm_mean_y

        elif norm == 2:
            y_pred = y_pred * (y_max - y_min) + y_min

        elif norm == 3:
            y_pred = y_pred * numpy.power(10, power)
        # ----------------------------------------------

        y_pred_all.append(y_pred)

        # -----------------------------------------------------------------------------------
        # ===================================================================================

    y_predicted = numpy.vstack(y_pred_all).T

    return y_predicted, y_test


def get_averaged_feature(pred_y, true_y, labels):
    """Return category-averaged features"""

    labels_set = numpy.unique(labels)

    pred_y_av = numpy.array([numpy.mean(pred_y[labels == c, :], axis=0) for c in labels_set])
    true_y_av = numpy.array([numpy.mean(true_y[labels == c, :], axis=0) for c in labels_set])

    return pred_y_av, true_y_av, labels_set


# ======================================================================================================================
# Running part

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

# Repeat (for different normalization):
# All return have average loss
# 1: none
# 2: none, z-score
# 3: none, z-score, min-max
# 4: none, z-score, min-max, decimal

repeat = 4
titles = ['No norm', 'Z-Score', 'Min-Max', 'Decimal Scaling']

norm_none, norm_z, norm_min_max, norm_decimal = data_prepare(dataset, regine_of_interest, image_feature,
                                                                           layers, voxel, repeat=repeat)

condition = 4

average_loss = []

loss_none = norm_none[1, :]
loss_z = norm_z[1, :]
loss_min_max = norm_min_max[1, :]
loss_decimal = norm_decimal[1, :]

average_loss = numpy.append(average_loss, avg_loss(loss_none), axis=0)
average_loss = numpy.append(average_loss, avg_loss(loss_z), axis=0)
average_loss = numpy.append(average_loss, avg_loss(loss_min_max), axis=0)
average_loss = numpy.append(average_loss, avg_loss(loss_decimal), axis=0)

plt.figure(figsize=(12, 12))
plt.suptitle('Prediction with different normalization')

plt.subplot(condition, 1, 1)
plt.title('No normalization')
plt.bar(range(1000), norm_none[0, :])

plt.subplot(condition, 1, 2)
plt.title('Z-score')
plt.bar(range(1000), norm_z[0, :])

plt.subplot(condition, 1, 3)
plt.title('Min-Max')
plt.bar(range(1000), norm_min_max[0, :])

plt.subplot(6, 1, 4)
plt.title('Decimal Scaling')
plt.bar(range(1000), norm_decimal[0, :])

average_loss = numpy.array(average_loss)
print('average loss: ', average_loss.shape)

plt.subplot(condition, 1, condition)
plt.title('Difference')
plt.bar(titles[:(repeat - 1)], average_loss)

plt.tight_layout(rect=[0, 0, 1, 0.99])
