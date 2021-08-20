import datetime
import os.path

import bdpy
import h5py
import numpy
import slir
from bdpy.ml import add_bias
from bdpy.preproc import select_top
from bdpy.stats import corrcoef
from matplotlib import pyplot as plt


# TODO: find the phenomenon of normalization cause algorithm understand the data or misunderstand the data
# TODO: find the noise increase or decrease through different normalization technics
# TODO: Noise precision β: means the distribution of noise. separate -> lower, gather -> higher

def generic_objective_decoding(data_all, img_feature, layer_all, voxel_all, norm_type):
    print('Data Preparing...')

    # ROIs setting ------------------
    subject_1 = 's1'
    roi_vc = ['VC', 'ROI_VC = 1']

    cor_voxel = voxel_all[roi_vc[0]]
    vc_mark = roi_vc[1]

    iterTimes = 200
    # -------------------------------

    print('Subject: %s, ROI: %s, Iteration: %s' % (subject_1, roi_vc[0], iterTimes))

    for layer in layer_all:
        data = data_all[subject_1]

        if norm_type == 0:
            print('No normalizing ==========================')
            none_all = get_result(data, img_feature, cor_voxel,
                                  vc_mark, layer, 0, iterTimes)

            return none_all

        elif norm_type == 1:
            print('Z-score normalization using =============')
            z_all = get_result(data, img_feature, cor_voxel,
                               vc_mark, layer, 1, iterTimes)

            return z_all

        elif norm_type == 2:
            print('Min-Max normalization using =============')
            min_max_all = get_result(data, img_feature, cor_voxel,
                                     vc_mark, layer, 2, iterTimes)

            return min_max_all

        elif norm_type == 3:
            print('Decimal scaling normalization using ==============')
            decimal_all = get_result(data, img_feature, cor_voxel,
                                     vc_mark, layer, 3, iterTimes)

            return decimal_all

        elif norm_type == 'all':

            print('No normalizing ==========================')
            none_all = get_result(data, img_feature, cor_voxel,
                                  vc_mark, layer, 0, iterTimes)

            print('Z-score normalization using =============')
            z_all = get_result(data, img_feature, cor_voxel,
                               vc_mark, layer, 1, iterTimes)

            print('Min-Max normalization using =============')
            min_max_all = get_result(data, img_feature, cor_voxel,
                                     vc_mark, layer, 2, iterTimes)

            print('Decimal scaling normalization using ==============')
            decimal_all = get_result(data, img_feature, cor_voxel,
                                     vc_mark, layer, 3, iterTimes)

            return none_all, z_all, min_max_all, decimal_all

        else:
            print('Error input type: norm_type. norm_type should be: [0, 1, 2, 3, all]')


# Algorithm part ==============================================================================
def get_result(sbj_1, img_feature, cor_voxels, vc, layer, norm_type: int, iter_times: int):
    # data separate -----------------------------
    x = sbj_1.select(vc)

    data_type = sbj_1.select('DataType')
    labels = sbj_1.select('stimulus_id')

    y = img_feature.select(layer)
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

    pred_y, true_y = algorithm_predict_feature(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                               num_voxel=cor_voxels, information=[layer],
                                               norm=norm_type, iter_num=iter_times)
    decimal_pred = pred_y
    decimal_true = true_y

    time_end = datetime.datetime.now()
    time_all = time_end - time_start

    print('Time cost: %s s' % time_all)

    time_all_seconds = time_all.seconds

    decimal_pred_pt = decimal_pred[i_pt, :]
    decimal_pred_im = decimal_pred[i_im, :]
    decimal_true_pt = decimal_true[i_pt, :]
    decimal_true_im = decimal_true[i_im, :]

    decimal_p_pt_av, decimal_t_pt_av, useless = get_averaged_feature(decimal_pred_pt,
                                                                     decimal_true_pt,
                                                                     test_label_pt)
    decimal_p_im_av, decimal_t_im_av, useless = get_averaged_feature(decimal_pred_im,
                                                                     decimal_true_im,
                                                                     test_label_im)

    output = {'pred_pt': decimal_pred_pt,
              'pred_im': decimal_pred_im,
              'true_pt': decimal_true_pt,
              'true_im': decimal_true_im,
              'p_pt_av': decimal_p_pt_av,
              't_pt_av': decimal_t_pt_av,
              'p_im_av': decimal_p_im_av,
              't_im_av': decimal_t_im_av,
              'time': time_all_seconds}

    save_to_hdf5(hdf5_dir=result_dir, layer=layer, tec=norm_type, iteration=iter_times, data_dict=output)

    return output


def algorithm_predict_feature(x_train, y_train, x_test, y_test,
                              num_voxel, information: list,
                              norm: (0, 1, 2, 3), iter_num: int):
    # Training iteration
    n_iter = iter_num

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

        power = 1

        while x_abs_max > 1:
            x_abs_max /= 10
            power += 1

        x_train = x_train / numpy.power(10, power)
        x_test = x_test / numpy.power(10, power)
    # ---------------------------------------------------

    # save predict value
    model = slir.SparseLinearRegression(n_iter=n_iter, prune_mode=1)
    y_pred_all = []

    # =========================================================================================
    # define the neural network architecture (convolutional net) ------------------------------

    for i in range(1000):

        print('Layer: %s, Image Feature: %d ##########################' %
              (information[0], i))

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
            y_pred = y_pred * (y_max - y_min) + y_min
            y_pred = y_pred / 2 + 1

        elif norm == 3:
            y_pred = y_pred * numpy.power(10, power)
        # ----------------------------------------------

        y_pred_all.append(y_pred)
        print('Finish ------------------------------')

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


# ---------------------------------------------------------------------------------------------

# Plot part ===========================================================================
def df_norm_avg_loss_plot(title):
    titles = numpy.array(['No norm', 'Z-Score', 'Min-Max', 'Decimal Scaling'])

    # Seen experiment ------------------------------
    n_pt_av = numpy.mean(none['p_pt_av'][0, :])
    z_pt_av = numpy.mean(z['p_pt_av'][0, :])
    m_pt_av = numpy.mean(min_max['p_pt_av'][0, :])
    d_pt_av = numpy.mean(decimal['p_pt_av'][0, :])

    t_pt_av = numpy.mean(none['t_pt_av'][0, :])

    n_pt_loss = t_pt_av - n_pt_av
    z_pt_loss = t_pt_av - z_pt_av
    m_pt_loss = t_pt_av - m_pt_av
    d_pt_loss = t_pt_av - d_pt_av
    # ----------------------------------------------

    # Imaginary part -------------------------------
    n_im_av = numpy.mean(none['p_im_av'][0, :])
    z_im_av = numpy.mean(z['p_im_av'][0, :])
    m_im_av = numpy.mean(min_max['p_im_av'][0, :])
    d_im_av = numpy.mean(decimal['p_im_av'][0, :])

    t_im_av = numpy.mean(none['t_im_av'][0, :])

    n_im_loss = t_im_av - n_im_av
    z_im_loss = t_im_av - z_im_av
    m_im_loss = t_im_av - m_im_av
    d_im_loss = t_im_av - d_im_av
    # ----------------------------------------------

    loss_pt_all = [n_pt_loss, z_pt_loss, m_pt_loss, d_pt_loss]
    loss_im_all = [n_im_loss, z_im_loss, m_im_loss, d_im_loss]

    plt.suptitle(title)
    plt.grid(True)

    plt.subplot(221)
    plt.title('Lose of seen images')
    plt.bar(titles, loss_pt_all)

    plt.subplot(222)
    plt.title('Lose of imaginary images')
    plt.bar(titles, loss_im_all)

    plt.subplot(223)
    plt.title('Lose of seen images - Limit Y')
    plt.ylim(-2, 4)
    plt.bar(titles, loss_pt_all)

    plt.subplot(224)
    plt.title('Lose of imaginary images - Limit Y')
    plt.ylim(-2, 4)
    plt.bar(titles, loss_im_all)


def norm_trainSet_contrast(title: str, pattern: int):
    # Compare original data and after normalization

    # Pattern:
    # 0 = get bias for each image feature
    # 1 = get the min & max value
    # 2 = Negative value occupation

    # Get x ---------------------------------
    data = dataset['s1']
    x = data.select(regine_of_interest['VC'])
    # ---------------------------------------

    # Get y -------------------------------------
    y = image_feature.select('cnn1')
    y_label = image_feature.select('ImageID')
    labels = data.select('stimulus_id')
    y_sort = bdpy.get_refdata(y, y_label, labels)
    # -------------------------------------------

    data_type = data.select('DataType')
    i_train = (data_type == 1).flatten()

    # Useful x & y ---------------------------------------------------
    x_train = x[i_train, :]
    y_train = y_sort[i_train, :]
    y_train_unit = y_train[:, 0]

    correlation = corrcoef(y_train_unit, x_train, var='col')

    x_train, voxel_index = select_top(x_train, numpy.abs(correlation),
                                      1000, axis=1,
                                      verbose=False)
    # ----------------------------------------------------------------

    # Normalization ---------------------------------
    norm_mean_x = numpy.mean(x_train, axis=0)
    norm_scale_x = numpy.std(x_train, axis=0, ddof=1)

    x_min = numpy.min(x_train)
    x_max = numpy.max(x_train)

    x_train_abs = numpy.abs(x_train)
    x_abs_max = numpy.max(x_train_abs)

    power = 1

    while x_abs_max > 1:
        x_abs_max /= 10
        power += 1
    # -----------------------------------------------

    # Normalization lists -----------------------------------------
    x_none = x_train[0, :]
    x_z = ((x_train - norm_mean_x) / norm_scale_x)[0, :]
    x_min_max = ((x_train - x_min) / (x_max - x_min) * 2 - 1)[0, :]
    x_decimal = (x_train / numpy.power(10, power))[0, :]
    # -------------------------------------------------------------

    # Min, Max value -----------------------------------
    n_min = x_min
    n_max = x_max

    z_min = numpy.min(x_z)
    z_max = numpy.max(x_z)

    m_min = numpy.min(x_min_max)
    m_max = numpy.max(x_min_max)

    d_min = numpy.min(x_decimal)
    d_max = numpy.max(x_decimal)

    max_list = numpy.array([n_max, z_max, m_max, d_max])
    min_list = numpy.array([n_min, z_min, m_min, d_min])
    # --------------------------------------------------

    # Negative Occupation -------------------------
    n_oc = numpy.abs(n_min) / n_max
    z_oc = numpy.abs(z_min) / z_max
    m_oc = numpy.abs(m_min) / m_max
    d_oc = numpy.abs(d_min) / d_max

    oc_list = numpy.array([n_oc, z_oc, m_oc, d_oc])
    # ---------------------------------------------

    plt.grid(True)

    if pattern == 0:
        # All image feature bias
        plt.suptitle(title)

        plt.subplot(411)
        plt.title('No normalization')
        plt.bar(range(1000), numpy.abs(x_none - x_none))

        plt.subplot(412)
        plt.title('Z-Score')
        plt.bar(range(1000), numpy.abs(x_none - x_z))

        plt.subplot(413)
        plt.title('Min-Max')
        plt.bar(range(1000), numpy.abs(x_none - x_min_max))

        plt.subplot(414)
        plt.title('Decimal Scaling')
        plt.bar(range(1000), numpy.abs(x_none - x_decimal))

    elif pattern == 1:
        # Min, Max values
        plt.subplot(121)
        plt.title('Largest & Smallest value')
        plt.bar(['None', 'Z-Score', 'Min-Max', 'Decimal Scaling'], max_list)
        plt.bar(['None', 'Z-Score', 'Min-Max', 'Decimal Scaling'], min_list)

        plt.subplot(122)
        plt.title('Largest & Smallest value - Limit Y')
        plt.ylim(-2, 4)
        plt.bar(['None', 'Z-Score', 'Min-Max', 'Decimal Scaling'], max_list)
        plt.bar(['None', 'Z-Score', 'Min-Max', 'Decimal Scaling'], min_list)

    elif pattern == 2:
        am_list = [neg_opt_value_ratio(x_none),
                   neg_opt_value_ratio(x_z),
                   neg_opt_value_ratio(x_min_max),
                   neg_opt_value_ratio(x_decimal)]

        plt.suptitle('Negative value occupation')

        plt.subplot(131)
        plt.title('Min value / Max value')
        plt.bar(['No normalization', 'Z-Score', 'Min-Max', 'Decimal Scaling'], oc_list)

        plt.subplot(132)
        plt.title('Number of Neg value / Number of Pos value')
        plt.bar(['No normalization', 'Z-Score', 'Min-Max', 'Decimal Scaling'], am_list)

        plt.subplot(133)
        plt.title('Number of Neg value / Number of Pos value - Y limit')
        plt.ylim(0, 0.4)
        plt.bar(['No normalization', 'Z-Score', 'Min-Max', 'Decimal Scaling'], am_list)

    else:
        print('Unknown pattern. (っ °Д °;)っ')


def neg_opt_value_ratio(x_list):
    neg = 0
    opt = 0

    for i in x_list:

        if i >= 0:
            opt += 1
        else:
            neg += 1

    return neg / opt


def norm_pred_contrast(title: str):
    # Compare predict data in each normalization with no normalization

    # Pattern:
    # 0 = separate seen & imaginary
    # 1 = combined seen & imaginary

    n_pt = none['pred_pt'][0, :]
    n_im = none['pred_im'][0, :]

    z_pt = z['pred_pt'][0, :]
    z_im = z['pred_im'][0, :]

    m_pt = min_max['pred_pt'][0, :]
    m_im = min_max['pred_im'][0, :]

    d_pt = decimal['pred_pt'][0, :]
    d_im = decimal['pred_im'][0, :]

    plt.suptitle(title + ' - Absolute Value')
    plt.grid(True)

    # No Normalization -------------------------
    plt.subplot(421)
    plt.title('No normalization Seen')
    plt.bar(range(1000), numpy.abs(n_pt - n_pt))

    plt.subplot(422)
    plt.title('No normalization Imaginary')
    plt.bar(range(1000), numpy.abs(n_im - n_im))
    # ------------------------------------------

    # Z-Score ----------------------------------
    plt.subplot(423)
    plt.title('Z-Score Seen')
    plt.bar(range(1000), numpy.abs(n_pt - z_pt))

    plt.subplot(424)
    plt.title('Z-Score Imaginary')
    plt.bar(range(1000), numpy.abs(n_im - z_im))
    # ------------------------------------------

    # Min-Max ----------------------------------
    plt.subplot(425)
    plt.title('Min-Max Seen')
    plt.bar(range(1000), numpy.abs(n_pt - m_pt))

    plt.subplot(426)
    plt.title('Min-Max Imaginary')
    plt.bar(range(1000), numpy.abs(n_im - m_im))
    # ------------------------------------------

    # Decimal Scaling --------------------------
    plt.subplot(427)
    plt.title('Decimal Scaling  Seen')
    plt.bar(range(1000), numpy.abs(n_pt - d_pt))

    plt.subplot(428)
    plt.title('Decimal Scaling  Seen')
    plt.bar(range(1000), numpy.abs(n_im - d_im))
    # ------------------------------------------


def time_plot():
    times_second = [none['time'], z['time'], min_max['time'], decimal['time']]
    labels = ['None', 'Z-SCore', 'Min-Max', 'Decimal Scaling']
    hours = numpy.array([])

    for i in range(0, len(times_second)):
        hour = times_second[i] / 3600
        hours = numpy.append(hours, hour)

    plt.title('Time Cost')
    plt.ylabel('Hours')
    plt.xlabel('Normalization Types')
    plt.bar(labels, hours)


# -------------------------------------------------------------------------------------

# Save result ==============================================================
def error_detected(name: str):
    content = 'Group |[{ %s }]| already exists, y/n continue? ' % name

    user_input = input(content).lower().strip()

    while True:
        if user_input == 'y':
            return True
        elif user_input == 'n':
            return False

        else:
            user_input = input('Unknown input. ' + content).lower().strip()


def save_to_hdf5(hdf5_dir: str, layer, tec: int, iteration: int, data_dict: dict):
    print('Save to results.hdf5 !!!!!!')

    # r:        Read only
    # r+:       Read / write, file must exist
    # w:        Create file, truncate if exists
    # w- / x:   Create file, fail if exists
    # a:        Read/write if exists, create otherwise
    dataTypes = ['pred_pt', 'pred_im', 'true_pt', 'true_im',
                 'p_pt_av', 't_pt_av', 'p_im_av', 't_im_av',
                 'time']

    norm_tec = ['none', 'z-score', 'min-max', 'decimal']

    hdf5 = h5py.File(hdf5_dir, 'r+')

    try:
        layer_group = hdf5.create_group(layer)
    except ValueError:
        print('Layer [%s] already exists, using it directly.' % layer)
        layer_group = hdf5[layer]

    try:
        target = layer_group.create_group('iter_' + str(iteration))
    except ValueError:
        print('Layer [%s] already exists, using it directly.' % layer)
        target = layer_group['iter_200']

    try:
        sub = target.create_group(norm_tec[tec])

        for dt in dataTypes:
            sub.create_dataset(dt, data=data_dict[dt])

        print('Data Collected. (。・∀・)ノ\n')

    except ValueError:
        if error_detected(norm_tec[tec]):
            sub = target.create_group(norm_tec[tec])

            for dt in dataTypes:
                sub.create_dataset(dt, data=data_dict[dt])

            print('Data Collected. (。・∀・)ノ\n')

        else:
            print('Data collection failed. Reason: User Cancelling. (＃°Д°)')

    hdf5.close()


# --------------------------------------------------------------------------

# Run Program ================================================================================

folder_dir = 'MSc Project\\bxz056\\data\\'
plot_result_dir = 'MSc Project\\bxz056\\plots\\results\\'
result_dir = 'G:\\Entrance\\Coding_Training\\PythonProgram\\MSc Project\\bxz056\\results.hdf5'

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

# norm_type (for different normalization):
#   0: none
#   1: z-score
#   2: min-max
#   3: decimal
#   all: all of them

none, z, min_max, decimal = generic_objective_decoding(dataset, image_feature, layers, voxel,
                                                       norm_type='all')
