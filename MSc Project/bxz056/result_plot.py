import os

import bdpy
import h5py
import numpy
import sklearn.metrics as sklearn
from bdpy.preproc import select_top
from bdpy.stats import corrcoef
from matplotlib import pyplot as plt


def time_plot(data, layer: str):
    loc = data[layer]
    i_keys = ['iter_50', 'iter_100', 'iter_150', 'iter_200']

    labels = ['None', 'Z-SCore', 'Min-Max', 'Decimal Scaling']

    plt.suptitle('Time Cost')

    i = 0
    for i_key in i_keys:
        i += 1

        n_t = loc[i_key]['none']['time']
        z_t = loc[i_key]['z-score']['time']
        m_t = loc[i_key]['min-max']['time']
        d_t = loc[i_key]['decimal']['time']

        times = numpy.array([n_t, z_t, m_t, d_t]) / 3600

        plt.subplot(2, 2, i)
        plt.title(i_key)
        plt.ylim(0, 3.5)
        plt.bar(labels, times, color='royalblue')


def norm_pred_contrast(title: str, none, z, min_max, decimal):
    # Compare predict data in each normalization with no normalization

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


# ----------------------------------------
def box_plot_pt(none, z, min_max, decimal):
    n_p_pt = none['pred_pt'][0, :]
    z_p_pt = z['pred_pt'][0, :]
    m_p_pt = min_max['pred_pt'][0, :]
    d_p_pt = decimal['pred_pt'][0, :]

    n_t_pt = none['true_pt'][0, :]
    z_t_pt = z['true_pt'][0, :]
    m_t_pt = min_max['true_pt'][0, :]
    d_t_pt = decimal['true_pt'][0, :]

    plt.suptitle('Box Plot - Seen Image Prediction & True')

    plt.subplot(181)
    plt.title('Prediction - No Normalization')
    plt.boxplot(n_p_pt, showmeans=True)

    plt.subplot(182)
    plt.title('True - No Normalization')
    plt.boxplot(n_t_pt, showmeans=True)

    plt.subplot(183)
    plt.title('Prediction - Z-Score')
    plt.boxplot(z_p_pt, showmeans=True)

    plt.subplot(184)
    plt.title('True - Z-Score')
    plt.boxplot(z_t_pt, showmeans=True)

    plt.subplot(185)
    plt.title('Prediction - Min-Max')
    plt.boxplot(m_p_pt, showmeans=True)

    plt.subplot(186)
    plt.title('True - Min-Max')
    plt.boxplot(m_t_pt, showmeans=True)

    plt.subplot(187)
    plt.title('Prediction - Decimal')
    plt.boxplot(d_p_pt, showmeans=True)

    plt.subplot(188)
    plt.title('True - Decimal')
    plt.boxplot(d_t_pt, showmeans=True)


def box_plot_im(none, z, min_max, decimal):
    n_p_im = none['pred_im'][0, :]
    z_p_im = z['pred_im'][0, :]
    m_p_im = min_max['pred_im'][0, :]
    d_p_im = decimal['pred_im'][0, :]

    n_t_im = none['true_im'][0, :]
    z_t_im = z['true_im'][0, :]
    m_t_im = min_max['true_im'][0, :]
    d_t_im = decimal['true_im'][0, :]

    plt.suptitle('Box Plot - Seen Image Prediction & True')

    plt.subplot(181)
    plt.title('Prediction - No Normalization')
    plt.boxplot(n_p_im, showmeans=True)

    plt.subplot(182)
    plt.title('True - No Normalization')
    plt.boxplot(n_t_im, showmeans=True)

    plt.subplot(183)
    plt.title('Prediction - Z-Score')
    plt.boxplot(z_p_im, showmeans=True)

    plt.subplot(184)
    plt.title('True - Z-Score')
    plt.boxplot(z_t_im, showmeans=True)

    plt.subplot(185)
    plt.title('Prediction - Min-Max')
    plt.boxplot(m_p_im, showmeans=True)

    plt.subplot(186)
    plt.title('True - Min-Max')
    plt.boxplot(m_t_im, showmeans=True)

    plt.subplot(187)
    plt.title('Prediction - Decimal')
    plt.boxplot(d_p_im, showmeans=True)

    plt.subplot(188)
    plt.title('True - Decimal')
    plt.boxplot(d_t_im, showmeans=True)


# ----------------------------------------

# -------------------------------------
def std_plot(results: dict, layer: str):
    test_dataset = read_xy_std_data(layer)[layer]
    loc = results[layer]['iter_200']

    labels = ['none', 'z-score', 'min-max', 'decimal']

    x_test_keys = ['n_test_std', 'z_test_std', 'm_test_std', 'd_test_std']
    y_keys = ['n_y_std', 'z_y_std', 'm_y_std', 'd_y_std']
    x_train_keys = ['n_train_std', 'z_train_std', 'm_train_std', 'd_train_std']

    x_test_std = numpy.array([])
    for k in x_test_keys:
        x_test_std = numpy.append(x_test_std, test_dataset[k])

    y_std = numpy.array([])
    for k in y_keys:
        y_std = numpy.append(y_std, test_dataset[k])

    pred_std = numpy.array([])
    for la in labels:
        pred_data = numpy.append(loc[la]['pred_pt'], loc[la]['pred_im'])
        pred_std = numpy.append(pred_std, numpy.mean(pred_data))

    x_train_std = numpy.array([])
    for tr in x_train_keys:
        x_train_std = numpy.append(x_train_std, test_dataset[tr])

    plt.suptitle('STD Difference')

    # Plot 1 ------------------------------------------------------
    plt.subplot(221)
    plt.title('fMRI Data - Test')
    plt.ylim(0, 1)
    plt.bar(labels, x_test_std, color='cornflowerblue')

    for x, y in zip(labels, x_test_std):
        plt.text(x, 0.05, '%.5f' % y, ha='center', va='bottom')
    # -------------------------------------------------------------

    # Plot 2 ------------------------------------------------------
    plt.subplot(222)
    plt.title('Prediction')
    plt.ylim(-2, 0)
    plt.bar(labels, pred_std, color='coral')

    for x, y in zip(labels, pred_std):
        plt.text(x, -0.15, '%.5f' % y, ha='center', va='bottom')
    # -------------------------------------------------------------

    # Plot 3 ------------------------------------------------------
    plt.subplot(223)
    plt.title('fMRI data - Training')
    plt.ylim(0, 1.2)
    plt.bar(labels, x_train_std, color='cornflowerblue')

    for x, y in zip(labels, x_train_std):
        plt.text(x, 0.05, '%.5f' % y, ha='center', va='bottom')
    # -------------------------------------------------------------

    # Plot 4 ------------------------------------------------------
    plt.subplot(224)
    plt.title('Image Feature')
    plt.ylim(0, 1.2)
    plt.bar(labels, y_std, color='coral')

    for x, y in zip(labels, y_std):
        plt.text(x, 0.05, '%.5f' % y, ha='center', va='bottom')
    # -------------------------------------------------------------


def create_xy_std_data(layer: str):
    print("Creating ['%s'] test data by normalization: [none, z-score, min-max, decimal]" % layer)

    file = h5py.File('G:\\Entrance\\Coding_Training\\PythonProgram\\MSc Project\\bxz056\\HDF5s\\xy_std_data.hdf5',
                     'a')

    # Labels -----------------------------
    s1, img = read_subject_1()
    labels = s1.select('stimulus_id')
    data_type = s1.select('DataType')

    i_train = (data_type == 1).flatten()

    i_test_pt = (data_type == 2).flatten()
    i_test_im = (data_type == 3).flatten()
    i_test = i_test_im + i_test_pt
    # ------------------------------------

    # Image Feature -----------------------------
    y = img.select(layer)
    y_label = img.select('ImageID')
    y_sort = bdpy.get_refdata(y, y_label, labels)

    y_train = y_sort[i_train, :]
    y_train_unit = y_train[:, 0]
    # -------------------------------------------

    # fMRI data -----------------------------------------------------------------------------------
    x = s1.select('ROI_VC = 1')
    x_train = x[i_train, :]
    x_test = x[i_test, :]
    print('x_test: ', x_test.shape)
    correlation = corrcoef(y_train_unit, x_train, var='col')

    print('correlated...')
    x_train, voxel_index = select_top(x_train, numpy.abs(correlation), 1000, axis=1, verbose=False)
    x_test = x_test[:, voxel_index]
    print('x_test: ', x_test.shape)
    # ---------------------------------------------------------------------------------------------

    print()

    # No Normalization ------
    n_test = x_test
    n_test_std = numpy.std(x_test)

    n_train = x_train
    n_train_std = numpy.std(x_train)

    n_y = y_train_unit
    n_y_std = numpy.std(n_y)

    print('n_test: ', n_test.shape)
    print('n_y: ', n_y.shape)
    # -----------------------

    # Z-Score -----------------------------------------
    norm_mean_x = numpy.mean(x_train, axis=0)
    norm_scale_x = numpy.std(x_train, axis=0, ddof=1)

    norm_mean_y = numpy.mean(y_train_unit, axis=0)
    std_y = numpy.std(y_train_unit, axis=0, ddof=1)
    norm_scale_y = 1 if std_y == 0 else std_y

    z_test = (x_test - norm_mean_x) / norm_scale_x
    z_test_std = numpy.std(z_test)

    z_train = (x_train - norm_mean_x) / norm_scale_x
    z_train_std = numpy.std(z_train)

    z_y = (y_train_unit - norm_mean_y) / norm_scale_y
    z_y_std = numpy.std(z_y)

    print('z_test: ', z_test.shape)
    print('z_y: ', z_y.shape)
    # -------------------------------------------------

    # Min-Max -----------------------------------------
    x_min = numpy.min(x_train)
    x_max = numpy.max(x_train)

    y_min = numpy.min(y_train_unit)
    y_max = numpy.max(y_train_unit)

    m_test = (x_test - x_min) / (x_max - x_min) * 2 - 1
    m_test_std = numpy.std(m_test)

    m_train = (x_train - x_min) / (x_max - x_min) * 2 - 1
    m_train_std = numpy.std(m_train)

    m_y = (y_train_unit - y_min) / (y_max - y_min) * 2 - 1
    m_y_std = numpy.std(m_y)

    print('m_test: ', m_test.shape)
    print('m_y: ', m_y.shape)
    # -------------------------------------------------

    # Decimal Scaling ----------------------
    x_train_abs = numpy.abs(x_train)
    x_abs_max = numpy.max(x_train_abs)

    power = 1
    while x_abs_max > 1:
        x_abs_max /= 10
        power += 1

    y_train_unit_abs = numpy.abs(y_train_unit)
    y_abs_max = numpy.max(y_train_unit_abs)

    power_y = 1
    while y_abs_max > 1:
        y_abs_max /= 10
        power_y += 1

    d_test = x_test / numpy.power(10, power)
    d_test_std = numpy.std(d_test)

    d_train = x_train / numpy.power(10, power)
    d_train_std = numpy.std(d_train)

    d_y = y_train_unit / numpy.power(10, power_y)
    d_y_std = numpy.std(d_y)

    print('d_test: ', d_test.shape)
    print('d_y: ', d_y.shape)
    # --------------------------------------

    labels = {'n_train': n_train, 'z_train': z_train, 'm_train': m_train, 'd_train': d_train,
              'n_train_std': n_train_std, 'z_train_std': z_train_std,
              'm_train_std': m_train_std, 'd_train_std': d_train_std,
              'n_test': n_test, 'z_test': z_test, 'm_test': m_test, 'd_test': d_test,
              'n_test_std': n_test_std, 'z_test_std': z_test_std,
              'm_test_std': m_test_std, 'd_test_std': d_test_std,
              'n_y': n_y, 'z_y': z_y, 'm_y': m_y, 'd_y': d_y,
              'n_y_std': n_y_std, 'z_y_std': z_y_std, 'm_y_std': m_y_std, 'd_y_std': d_y_std}

    try:
        target = file.create_group(layer)
    except ValueError:
        target = file[layer]

    for lab in labels:
        try:
            target.create_dataset(lab, data=numpy.array(labels[lab]))

        except RuntimeError:
            del target[lab]
            target.create_dataset(lab, data=numpy.array(labels[lab]))

    file.close()


def read_xy_std_data(layer: str):
    path = 'G:\\Entrance\\Coding_Training\\PythonProgram\\MSc Project\\bxz056\\HDF5s\\xy_std_data.hdf5'
    file = h5py.File(path, 'r')
    group = file[layer]

    keys = list(group.keys())

    output = {}
    data = {}

    for k in keys:
        data[k] = numpy.array(group[k])

    output[layer] = data

    file.close()
    return output


# -------------------------------------


def mse_plot(data):
    norm = ['none', 'z-score', 'min-max', 'decimal']

    i = 0
    plt.suptitle('MSE - for Different ')

    for n in norm:
        i += 1
        pred_pt = data['cnn1']['iter_200'][n]['pred_pt']
        true_pt = data['cnn1']['iter_200'][n]['true_pt']

        pred_im = data['cnn1']['iter_200'][n]['pred_im']
        true_im = data['cnn1']['iter_200'][n]['true_im']

        pt_mse = sklearn.mean_squared_error(true_pt, pred_pt)
        im_mse = sklearn.mean_squared_error(true_im, pred_im)

        print('pt_mse: ', pt_mse)
        print('im_mse: ', im_mse)

        mse_list = numpy.array([pt_mse, im_mse])

        plt.subplot(1, 4, i)
        plt.title(n)
        plt.ylim(0, 16000)
        plt.bar(['Seen', 'Imaginary'], mse_list, color=['cornflowerblue', 'coral'])

        for x, y in zip(['Seen', 'Imaginary'], mse_list):
            plt.text(x, 0.05, '%.5f' % y, ha='center', va='bottom')


# ---------------------------------------------------
def norm_trainSet_contrast(title: str, pattern: int):
    # Compare original data and after normalization

    # Pattern:
    # 0 = get bias for each image feature
    # 1 = get the min & max value
    # 2 = Negative value occupation

    # Get x ---------------------------------
    data, image_feature = read_subject_1()
    x = data.select('ROI_VC = 1')
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


def read_subject_1():
    print('Read subject data')

    folder_dir = 'G:\\Entrance\\Coding_Training\\PythonProgram\\MSc Project\\bxz056\\data\\'

    path = {'s1': os.path.abspath(folder_dir + 'Subject1.h5'),
            'imageFeature': os.path.abspath(folder_dir + 'ImageFeatures.h5')}

    s1 = bdpy.BData(path['s1'])
    img = bdpy.BData(path['imageFeature'])

    return s1, img


def neg_opt_value_ratio(x_list):
    neg = 0
    opt = 0

    for i in x_list:

        if i >= 0:
            opt += 1
        else:
            neg += 1

    return neg / opt


# ---------------------------------------------------
def merged_av_plot(data: dict, layer: str, iteration: int, data_type: str, pattern: int):
    loc = data[layer]['iter_' + str(iteration)]

    n_keys = ['none', 'z-score', 'min-max', 'decimal']
    d_keys = ['cor_pt_av', 'cor_im_av', 'cor_cat_pt_av', 'cor_cat_im_av']

    i = 0
    plt.suptitle('Correlation for Merged Result')

    for n_key in n_keys:
        i += 1

        cor_pt_av = loc[n_key][d_keys[0]].reshape(50)
        cor_im_av = loc[n_key][d_keys[1]].reshape(50)
        cor_cat_pt_av = loc[n_key][d_keys[2]].reshape(50)
        cor_cat_im_av = loc[n_key][d_keys[3]].reshape(50)

        mean_pt = numpy.mean(cor_pt_av)
        mean_im = numpy.mean(cor_im_av)
        mean_cat_pt = numpy.mean(cor_cat_pt_av)
        mean_cat_im = numpy.mean(cor_cat_im_av)

        if data_type == 'tru':
            dataset = [cor_pt_av, cor_im_av]
            mean_set = [mean_pt, mean_im]
            x_label = ['mean_pt', 'mean_im']
            mean_title = '[Pred - True]'
            y_lim = 0.055

        elif data_type == 'cat':
            dataset = [cor_cat_pt_av, cor_cat_im_av]
            mean_set = [mean_cat_pt, mean_cat_im]
            x_label = ['mean_cat_pt', 'mean_cat_im']
            mean_title = '[Pred - Categories]'
            y_lim = 0.125

        else:
            print("Unknown data type.（*゜ー゜*）")
            break

        if pattern == 0:
            plt.subplot(2, 4, i)
            plt.title(n_key + ' - Seen ' + mean_title)
            plt.bar(range(50), dataset[0], color='royalblue')

            i += 1

            plt.subplot(2, 4, i)
            plt.title(n_key + ' - Imaginary ' + mean_title)
            plt.bar(range(50), dataset[1], color='coral')

        elif pattern == 1:
            plt.subplot(1, 4, i)
            plt.title(n_key + ' - ' + mean_title)
            plt.ylim(0, y_lim)
            plt.bar(x_label, mean_set, color=['royalblue', 'coral'])

        else:
            print('Unknown pattern. (´･ω･`)?')
            break


# =======================
def read_data(path: str):
    print('Getting File...')

    file = h5py.File(path, 'r')

    output = {}
    layer_list = list(file.keys())

    for layer in layer_list:
        iter_list = list(file[layer].keys())

        # collect iterations
        iter_dict = {}
        for iteration in iter_list:
            norm_list = list(file[layer][iteration].keys())

            # collect normalizations
            norm_dict = {}
            for norm in norm_list:
                target = file[layer][iteration][norm]
                type_list = list(target.keys())

                data_dict = {}
                for data_type in type_list:
                    data_dict[data_type] = numpy.array(target[data_type])

                norm_dict[norm] = data_dict
            iter_dict[iteration] = norm_dict
        output[layer] = iter_dict

    file.close()

    return output


def read_cat(path: str):
    print('Getting categories.')
    file = h5py.File(path, 'r')

    keys = ['cat_pt_av', 'cat_im_av', 'cat_pt_label', 'cat_im_label']

    output = {}

    for key in keys:
        output[key] = numpy.array(file[key])

    return output


# =======================

# -------------------------------------------------------------------------------------
data_dir = 'G:\\Entrance\\Coding_Training\\PythonProgram\\MSc Project\\bxz056\\HDF5s\\'

result_file = data_dir + 'results.hdf5'
cat_file = data_dir + 'cat_data.hdf5'
merge_file = data_dir + 'mergedResult.hdf5'

result = read_data(result_file)
merged = read_data(merge_file)
category = read_cat(cat_file)
