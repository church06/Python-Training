import datetime
import os.path

import bdpy
import h5py
import numpy
import slir
from bdpy.ml import add_bias
from bdpy.preproc import select_top
from bdpy.stats import corrcoef

import Tools


# TODO: find the phenomenon of normalization cause algorithm understand the data or misunderstand the data
# TODO: find the noise increase or decrease through different normalization technics
# TODO: Noise precision β: means the distribution of noise. separate -> lower, gather -> higher

def generic_objective_decoding(data_all, img_feature, R, S: str, L: str, N):
    print('Data Preparing...')

    subject = S
    data = data_all[subject]

    if N == 0:
        none_all = get_result(data, img_feature, S, R, L, 0)
        return none_all

    elif N == 1:
        z_all = get_result(data, img_feature, S, R, L, 1)
        return z_all

    elif N == 2:
        min_max_all = get_result(data, img_feature, S, R, L, 2)
        return min_max_all

    elif N == 3:
        decimal_all = get_result(data, img_feature, S, R, L, 3)
        return decimal_all

    elif N == 'all':
        get_result(data, img_feature, S, R, L, 1)
        get_result(data, img_feature, S, R, L, 2)
        get_result(data, img_feature, S, R, L, 3)
        get_result(data, img_feature, S, R, L, 0)
    else:
        print('Error input type: norm_type. norm_type should be: [0, 1, 2, 3, all]')


# Algorithm part ==============================================================================
def get_result(S_data, img_feature, S: str, R, L, N: int):
    tool = Tools.Tool()
    roi_s = {'VC': 'ROI_VC = 1', 'LVC': 'ROI_LVC = 1', 'HVC': 'ROI_HVC = 1',
             'V1': 'ROI_V1 = 1', 'V2': 'ROI_V2 = 1', 'V3': 'ROI_V3 = 1',
             'V4': 'ROI_V4 = 1',
             'LOC': 'ROI_LOC = 1', 'FFA': 'ROI_FFA = 1', 'PPA': 'ROI_PPA = 1'}

    voxel = {'VC': 1000, 'LVC': 1000, 'HVC': 1000,
             'V1': 500, 'V2': 500, 'V3': 500,
             'V4': 500,
             'LOC': 500, 'FFA': 500, 'PPA': 500}

    # data separate -----------------------------
    cor_voxel = voxel[R]
    roi_mark = roi_s[R]
    x = S_data.select(roi_mark)

    data_type = S_data.select('DataType')
    labels = S_data.select('stimulus_id')

    y = img_feature.select(L)
    y_label = img_feature.select('ImageID')
    y_sort = bdpy.get_refdata(y, y_label, labels)

    i_train = (data_type == 1).flatten()
    i_test_seen = (data_type == 2).flatten()
    i_test_img = (data_type == 3).flatten()
    i_test = i_test_img + i_test_seen
    # -------------------------------------------

    # get data & voxels --------------
    x_train = x[i_train, :]
    x_test = x[i_test, :]

    y_train = y_sort[i_train, :]
    y_test = y_sort[i_test, :]
    # --------------------------------

    # for test program ----------------
    # test_index = 2
    #
    # x_train = x_train[:test_index, :]
    # y_train = y_train[:test_index, :]
    # ---------------------------------

    # Separate imaginary & seen----------------------------------------
    i_pt = i_test_seen[i_test]  # Index for perception test within test
    i_im = i_test_img[i_test]  # Index for imagery test within test
    # -----------------------------------------------------------------

    # Get averaged predicted feature
    test_label_pt = labels[i_test_seen, :].flatten()
    test_label_im = labels[i_test_img, :].flatten()

    time_start = datetime.datetime.now()

    pred_y, true_y, a_list, w_list, g_list = algorithm_predict_feature(x_train=x_train, y_train=y_train,
                                                                       x_test=x_test, y_test=y_test,
                                                                       num_voxel=cor_voxel, info=[S, R, L, N],
                                                                       N=N)

    time_end = datetime.datetime.now()
    time_all = time_end - time_start

    print('Time cost: %s s' % time_all)

    time_all_seconds = time_all.seconds

    pred_pt = pred_y[i_pt, :]
    pred_im = pred_y[i_im, :]
    true_pt = true_y[i_pt, :]
    true_im = true_y[i_im, :]

    p_pt_av, t_pt_av, useless = get_averaged_feature(pred_pt, true_pt,
                                                     test_label_pt)
    p_im_av, t_im_av, useless = get_averaged_feature(pred_im, true_im,
                                                     test_label_im)

    output = {'pred_pt': pred_pt, 'pred_im': pred_im,
              'true_pt': true_pt, 'true_im': true_im,
              'p_pt_av': p_pt_av, 't_pt_av': t_pt_av,
              'p_im_av': p_im_av, 't_im_av': t_im_av,
              'time': time_all_seconds,
              'alpha': a_list, 'weight': w_list, 'gain': g_list}

    tool.save_to_result(S=S, L=L, R=R, N=N, data_dict=output)
    return output


def algorithm_predict_feature(x_train, y_train, x_test, y_test,
                              num_voxel, info: list,
                              N: int):
    # Training iteration
    n_unit = y_train.shape[1]
    n_iter = 200

    # Normalize brian data (x) --------------------------
    if N == 0:
        x_train = x_train
        x_test = x_test

    elif N == 1:
        # Z-Score
        norm_mean_x = numpy.mean(x_train, axis=0)
        norm_scale_x = numpy.std(x_train, axis=0, ddof=1)

        x_train = (x_train - norm_mean_x) / norm_scale_x
        x_test = (x_test - norm_mean_x) / norm_scale_x

    elif N == 2:
        # Min-Max
        x_min = numpy.min(x_train)
        x_max = numpy.max(x_train)

        x_train = (x_train - x_min) / (x_max - x_min) * 2 - 1
        x_test = (x_test - x_min) / (x_max - x_min) * 2 - 1

    elif N == 3:
        # Decimal Scaling
        x_train_abs = numpy.abs(x_train)
        x_abs_max = numpy.max(x_train_abs)

        power = 1
        while x_abs_max > 1:
            x_abs_max /= 10
            power += 1

        x_train = x_train / numpy.power(10, power)
        x_test = x_test / numpy.power(10, power)
    # ---------------------------------------------------

    # save predict value
    a_list, w_list, g_list = numpy.array([]), numpy.array([]), numpy.array([])
    y_pred_all = []

    # =========================================================================================
    # define the neural network architecture (convolutional net) ------------------------------

    for i in range(n_unit):
        print('===================================================================')
        print('Subject: %s, ROI: %s, Layer: %s, Norm tec: %s, Image Feature: %03d' %
              (info[0], info[1], info[2], info[3], i))

        print('Get single y unit.')
        # Image feature normalization - for all data
        y_train_unit = y_train[:, i]
        y_test_unit = y_test[:, i]
        # ------------------------------------------

        print('Compute normalizing parameters ------')
        # Normalization of image feature -----------------------------

        print(' 1. Z-Score')
        # Z-Score -----------------------------------
        norm_mean_y = numpy.mean(y_train_unit, axis=0)
        std_y = numpy.std(y_train_unit, axis=0, ddof=1)
        norm_scale_y = 1 if std_y == 0 else std_y
        # ---------------------------------------------

        print(' 2. Min-Max')
        # Min-Max ----------------
        y_min = numpy.min(y_train_unit)
        y_max = numpy.max(y_train_unit)
        # ------------------------

        print(' 3. Decimal Scaling')
        # Decimal Scaling ----------------
        y_train_unit_abs = numpy.abs(y_train_unit)
        y_abs_max = numpy.max(y_train_unit_abs)

        power = 1
        while y_abs_max > 1:
            y_abs_max /= 10
            power += 1
        # --------------------------------

        print('Do normalizing ----------------------')
        if N == 0:
            # No normalization
            y_train_unit = y_train_unit

        elif N == 1:
            # z-score
            y_train_unit = (y_train_unit - norm_mean_y) / norm_scale_y

        elif N == 2:
            # min-max
            y_train_unit = (y_train_unit - y_min) / (y_max - y_min) * 2 - 1

        elif N == 3:
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

        model = slir.SparseLinearRegression(n_iter=n_iter, prune_mode=1)

        # Training and test
        try:
            print('Fitting model...')
            model.fit(x_train, y_train_unit)  # Training

            print('Model predicting...')
            y_pred = model.predict(x_test)  # Test

            print('Collect GD data...')
            if i == 0:
                a_list, w_list, g_list = model.GD_data()

        except Exception as e:
            print(e)
            y_pred = numpy.zeros(y_test_unit.shape)

        # Denormalize ----------------------------------
        if N == 1:
            y_pred = y_pred * norm_scale_y + norm_mean_y

        elif N == 2:
            y_pred = (y_pred + 1) / 2
            y_pred = y_pred / (y_max - y_min) + y_min

        elif N == 3:
            y_pred = y_pred * numpy.power(10, power)
        # ----------------------------------------------

        y_pred_all.append(numpy.array(y_pred))
        print('Finish ------------------------------')
        print('===================================================================')

        # -----------------------------------------------------------------------------------
        # ===================================================================================

    y_predicted = numpy.vstack(y_pred_all).T

    return y_predicted, y_test, a_list, w_list, g_list


def get_averaged_feature(pred_y, true_y, labels):
    # Return category-averaged features

    labels_set = numpy.unique(labels)

    pred_y_av = numpy.array([numpy.mean(pred_y[labels == c, :], axis=0) for c in labels_set])
    true_y_av = numpy.array([numpy.mean(true_y[labels == c, :], axis=0) for c in labels_set])

    return pred_y_av, true_y_av, labels_set


# Run Program ================================================================================
folder_dir = 'data\\'

subjects = {'s1': os.path.abspath(folder_dir + 'Subject1.h5'),
            's2': os.path.abspath(folder_dir + 'Subject2.h5'),
            's3': os.path.abspath(folder_dir + 'Subject3.h5'),
            's4': os.path.abspath(folder_dir + 'Subject4.h5'),
            's5': os.path.abspath(folder_dir + 'Subject5.h5'),
            'imageFeature': os.path.abspath(folder_dir + 'ImageFeatures.h5')}

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

    # Subject 1 ~ 5
    if person != 'imageFeature':
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

# ---------------------------------------------------------------

targets = {'V1': 'ROI_V1 = 1', 'V2': 'ROI_V2 = 1', 'V3': 'ROI_V3 = 1',
           'V4': 'ROI_V4 = 1',
           'LOC': 'ROI_LOC = 1', 'FFA': 'ROI_FFA = 1', 'PPA': 'ROI_PPA = 1'}

# norm_type (for different normalization):
#   0:     none
#   1:     z-score
#   2:     min-max
#   3:     decimal
#   all:   all of them (no return)

for layer in ['cnn2', 'cnn4', 'cnn6', 'cnn8']:
    for t_roi in ['LOC', 'FFA', 'PPA']:
        generic_objective_decoding(data_all=dataset,
                                   img_feature=image_feature,
                                   R=t_roi, S='s1', L=layer,
                                   N='all')
