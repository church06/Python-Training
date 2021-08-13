import datetime
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

def data_prepare(subject, rois, img_feature, layer_all, voxel_all, norm_type):
    print('Data prepare:')
    print('-----------------')

    # Parameters  ========================================================

    non_pred = []
    non_true = []

    z_pred = []
    z_true = []

    min_max_pred = []
    min_max_true = []

    decimal_pred = []
    decimal_true = []

    # times
    non_time = 0
    z_time = 0
    min_max_time = 0
    decimal_time = 0

    # Parameters ========================================================

    # ROIs setting ------------------
    roi_single = {'VC': 'ROI_VC = 1'}
    layer_single = ['cnn1']
    subject_single = ['s1']
    # -------------------------------

    for sbj, roi, layer in product(subject_single, roi_single, layer_single):
        data = subject[sbj]
        x = data.select(rois[roi])

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

        # for test -----------------------
        # test_index = 2
        #
        # x_train = x_train[:test_index, :]
        # y_train = y_train[:test_index, :]
        # --------------------------------

        for i in range(norm_type):

            time_start = datetime.datetime.now()

            if i == 0:
                print('No normalizing =============================')
                pred_y, true_y = algorithm_predict_feature(x_train=x_train, y_train=y_train,
                                                           x_test=x_test, y_test=y_test,
                                                           num_voxel=voxel_all[roi], information=[sbj, roi, layer],
                                                           norm=i)
                non_pred = pred_y
                non_true = true_y

                time_end = datetime.datetime.now()
                non_time = time_end - time_start
                print('Time cost: %s s' % non_time)

            elif i == 1:
                print('Z-score normalization using ================')
                pred_y, true_y = algorithm_predict_feature(x_train=x_train, y_train=y_train,
                                                           x_test=x_test, y_test=y_test,
                                                           num_voxel=voxel_all[roi], information=[sbj, roi, layer],
                                                           norm=i)
                z_pred = pred_y
                z_true = true_y

                time_end = datetime.datetime.now()
                z_time = time_end - time_start
                print('Time cost: %s s' % z_time)

            elif i == 2:
                print('Min-Max normalization using ================')
                pred_y, true_y = algorithm_predict_feature(x_train=x_train, y_train=y_train,
                                                           x_test=x_test, y_test=y_test,
                                                           num_voxel=voxel_all[roi], information=[sbj, roi, layer],
                                                           norm=i)
                min_max_pred = pred_y
                min_max_true = true_y

                time_end = datetime.datetime.now()
                min_max_time = time_end - time_start
                print('Time cost: %s s' % min_max_time)

            elif i == 3:
                print('Decimal scaling normalization using =================')
                pred_y, true_y = algorithm_predict_feature(x_train=x_train, y_train=y_train,
                                                           x_test=x_test, y_test=y_test,
                                                           num_voxel=voxel_all[roi], information=[sbj, roi, layer],
                                                           norm=i)
                decimal_pred = pred_y
                decimal_true = true_y

                time_end = datetime.datetime.now()
                decimal_time = time_end - time_start

                print('Time cost: %s s' % decimal_time)

        # Separate imaginary & seen----------------------------------------
        i_pt = i_test_seen[i_test]  # Index for perception test within test
        i_im = i_test_img[i_test]  # Index for imagery test within test
        # -----------------------------------------------------------------

        # Datasets for imaginary & seen--------
        non_pred = numpy.array(non_pred)
        non_true = numpy.array(non_true)

        print('non_pred: ', non_pred.shape)
        print('non_true: ', non_true.shape)
        print('i_pt: ', i_pt.shape)
        print('i_im: ', i_im.shape)

        non_pred_pt = non_pred[i_pt, :]
        non_pred_im = non_pred[i_im, :]
        non_true_pt = non_true[i_pt, :]
        non_true_im = non_true[i_im, :]

        z_pred_pt = z_pred[i_pt, :]
        z_pred_im = z_pred[i_im, :]
        z_true_pt = z_true[i_pt, :]
        z_true_im = z_true[i_im, :]

        min_max_pred_pt = min_max_pred[i_pt, :]
        min_max_pred_im = min_max_pred[i_im, :]
        min_max_true_pt = min_max_true[i_pt, :]
        min_max_true_im = min_max_true[i_im, :]

        decimal_pred_pt = decimal_pred[i_pt, :]
        decimal_pred_im = decimal_pred[i_im, :]
        decimal_true_pt = decimal_true[i_pt, :]
        decimal_true_im = decimal_true[i_im, :]

        # -------------------------------------

        # Get averaged predicted feature
        test_label_pt = labels[i_test_seen, :].flatten()
        test_label_im = labels[i_test_img, :].flatten()

        # Get average feature ------------------------------------------------------------

        # No normalization --------------
        print('non pred pt: ', non_pred_pt.shape)
        print('non true pt: ', non_true_pt.shape)
        print('test label pt: ', test_label_pt.shape)
        print('test label im: ', test_label_im.shape)

        non_p_pt_av, non_t_pt_av, useless = get_averaged_feature(non_pred_pt,
                                                                 non_true_pt,
                                                                 test_label_pt)
        non_p_im_av, non_t_im_av, useless = get_averaged_feature(non_pred_im,
                                                                 non_true_im,
                                                                 test_label_im)

        # Z-Score Normalization ---------------------
        pred_y_pt_av, true_y_pt_av, test_label_set_pt = get_averaged_feature(z_pred_pt,
                                                                             z_true_pt,
                                                                             test_label_pt)
        pred_y_im_av, true_y_im_av, test_label_set_im = get_averaged_feature(z_pred_im,
                                                                             z_true_im,
                                                                             test_label_im)
        z_p_pt_av, z_t_pt_av = pred_y_pt_av, true_y_pt_av
        z_p_im_av, z_t_im_av = pred_y_im_av, true_y_im_av

        # Min-Max Normalization -----------------
        min_max_p_pt_av, min_max_t_pt_av, useless = get_averaged_feature(min_max_pred_pt,
                                                                         min_max_true_pt,
                                                                         test_label_pt)
        min_max_p_im_av, min_max_t_im_av, useless = get_averaged_feature(min_max_pred_im,
                                                                         min_max_true_im,
                                                                         test_label_im)

        # Decimal Scaling -----------------------
        decimal_p_pt_av, decimal_t_pt_av, useless = get_averaged_feature(decimal_pred_pt,
                                                                         decimal_true_pt,
                                                                         test_label_pt)
        decimal_p_im_av, decimal_t_im_av, useless = get_averaged_feature(decimal_pred_im,
                                                                         decimal_true_im,
                                                                         test_label_im)
        # ---------------------------------------------------------------------------------

        # Gather output --------------------------
        none_all = {'pred_pt': non_pred_pt,
                    'pred_im': non_pred_im,
                    'true_pt': non_true_pt,
                    'true_im': non_true_im,
                    'p_pt_av': non_p_pt_av,
                    't_pt_av': non_t_pt_av,
                    'p_im_av': non_p_im_av,
                    't_im_av': non_t_im_av,
                    'time': non_time}

        z_all = {'pred_pt': z_pred_pt,
                 'pred_im': z_pred_im,
                 'true_pt': z_true_pt,
                 'true_im': z_true_im,
                 'p_pt_av': z_p_pt_av,
                 't_pt_av': z_t_pt_av,
                 'p_im_av': z_p_im_av,
                 't_im_av': z_t_im_av,
                 'time': z_time}

        min_max_all = {'pred_pt': min_max_pred_pt,
                       'pred_im': min_max_pred_im,
                       'true_pt': min_max_true_pt,
                       'true_im': min_max_true_im,
                       'p_pt_av': min_max_p_pt_av,
                       't_pt_av': min_max_t_pt_av,
                       'p_im_av': min_max_p_im_av,
                       't_im_av': min_max_t_im_av,
                       'time': min_max_time}

        decimal_all = {'pred_pt': decimal_pred_pt,
                       'pred_im': decimal_pred_im,
                       'true_pt': decimal_true_pt,
                       'true_im': decimal_true_im,
                       'p_pt_av': decimal_p_pt_av,
                       't_pt_av': decimal_t_pt_av,
                       'p_im_av': decimal_p_im_av,
                       't_im_av': decimal_t_im_av,
                       'time': decimal_time}
        # ----------------------------------------

        print('Predict average seen feature: ', pred_y_pt_av.shape)
        print('True average seen feature: ', true_y_pt_av.shape)
        print('Predict average imaginary feature: ', pred_y_im_av.shape)
        print('True average imaginary feature: ', true_y_im_av.shape)

        # ----------------------------------------------------- Return setting
        if norm_type == 1:
            return none_all
        elif norm_type == 2:
            return none_all, z_all
        elif norm_type == 3:
            return none_all, z_all, min_max_all
        elif norm_type == 4:
            return none_all, z_all, min_max_all, decimal_all

        # --------------------------------------------------------------------

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
                                'true_feature': [z_true_pt, z_true_im],
                                'predicted_feature': [z_pred_pt, z_pred_im],
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


def algorithm_predict_feature(x_train, y_train, x_test, y_test, num_voxel, information: list, norm: (0, 1, 2, 3)):
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

        x_train = (x_train - x_min) / (x_max - x_min) * 2 - 1
        x_test = (x_test - x_min) / (x_max - x_min) * 2 - 1

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

        print('Subject: %s, Roi: %s, Layer: %s, Voxel: %d ##########################' %
              (information[0], information[1], information[2], i))

        print('Pick y units %s.' % i)
        # Image feature normalization - for all data
        y_train_unit = y_train[:, i]
        y_test_unit = y_test[:, i]
        # ------------------------------------------

        print('Compute normalizing parameters-------')
        # Normalization of image feature -----------------------------

        print('1. Z-Score')
        # Z-Score -----------------------------------
        norm_mean_y = numpy.mean(y_train_unit, axis=0)
        std_y = numpy.std(y_train_unit, axis=0, ddof=1)
        norm_scale_y = 1 if std_y == 0 else std_y
        # ---------------------------------------------

        print('2. Min-Max')
        # Min-Max ----------------
        y_min = numpy.min(y_train_unit)
        y_max = numpy.max(y_train_unit)
        # ------------------------

        print('3. Decimal Scaling')
        # Decimal Scaling ----------------
        y_train_unit_abs = numpy.abs(y_train_unit)
        y_abs_max = numpy.max(y_train_unit_abs)

        power = 1
        while y_abs_max > 1:
            y_abs_max /= 10
            power += 1
        # --------------------------------

        print('Do normalizing ----------------------')
        if norm == 0:
            # No normalization
            y_train_unit = y_train_unit
            y_test_unit = y_test_unit

        elif norm == 1:
            # z-score
            y_train_unit = (y_train_unit - norm_mean_y) / norm_scale_y

        elif norm == 2:
            # min-max
            y_train_unit = (y_train_unit - y_min) / (y_max - y_min) * 2 - 1

        elif norm == 3:
            # Decimal Scaling
            y_train_unit = y_train_unit / numpy.power(10, power)
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
            print('Fitting model...')
            model.fit(x_train, y_train_unit)  # Training

            print('Model predicting...')
            y_pred = model.predict(x_test)  # Test

        except Exception as e:
            print(e)
            y_pred = numpy.zeros(y_test_unit.shape)

        # Denormalize ----------------------------------
        if norm == 1:
            y_pred = y_pred * norm_scale_y + norm_mean_y

        elif norm == 2:
            y_pred = y_pred / 2 + 1
            y_pred = y_pred * (y_max - y_min) + y_min

        elif norm == 3:
            y_pred = y_pred * numpy.power(10, power)
        # ----------------------------------------------

        y_pred_all.append(y_pred)
        print('Finish-------------------------------')

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


def plot_pred_true_only(norm_pattern, title):
    plt.suptitle(title)

    plt.subplot(3, 1, 1)
    plt.title('Predict image feature')
    plt.bar(range(1000), norm_pattern[0, :])

    plt.subplot(3, 1, 2)
    plt.title('True image feature')
    plt.bar(range(1000), norm_pattern[2, :])

    bias = numpy.abs(norm_pattern[2, :] - norm_pattern[0, :])

    plt.subplot(3, 1, 3)
    plt.title('Bias')
    plt.plot(range(1000), bias)


def df_norm_avg_bias_plot(loss_list, title):
    titles = numpy.array(['No norm', 'Z-Score', 'Min-Max', 'Decimal Scaling'])

    loss_list = numpy.abs(loss_list)

    plt.suptitle(title)
    plt.bar(titles, loss_list)
    plt.tight_layout()


def sp_contrast(list_1, list_2, title, subtitle_1, subtitle_2):
    plt.suptitle(title)

    plt.subplot(311)
    plt.title(subtitle_1)
    plt.bar(range(1000), list_1[0, :])

    plt.subplot(312)
    plt.title(subtitle_2)
    plt.bar(range(1000), list_2[0, :])

    difference = numpy.abs(list_1[0, :] - list_2[0, :])

    plt.subplot(313)
    plt.title('Absolute Difference')
    plt.plot(range(1000), difference)


# ##########################################################################
# ==========================================================================
# Running part

folder_dir = 'MSc Project\\bxz056\\data\\'

subjects = {'s1': os.path.abspath(folder_dir + 'Subject1.h5'),
            's2': os.path.abspath(folder_dir + 'Subject2.h5'),
            's3': os.path.abspath(folder_dir + 'Subject3.h5'),
            's4': os.path.abspath(folder_dir + 'Subject4.h5'),
            's5': os.path.abspath(folder_dir + 'Subject5.h5'),
            'imageFeature': os.path.abspath(folder_dir + 'ImageFeatures.h5')}

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
none, z, min_max, decimal = data_prepare(dataset,
                                         regine_of_interest,
                                         image_feature,
                                         layers,
                                         voxel, norm_type=repeat)
