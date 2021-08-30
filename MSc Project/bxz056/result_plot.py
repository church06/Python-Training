import bdpy
import numpy
import sklearn.metrics as sklearn
from bdpy.preproc import select_top
from bdpy.stats import corrcoef
from matplotlib import pyplot as plt
import Tools


def time_plot(data: dict, layer: str):
    roi_s = list(data.keys())

    i = 0
    for roi in roi_s:
        i += 1
        loc = data[roi][layer]
        labels = ['None', 'Z-SCore', 'Min-Max', 'Decimal Scaling']

        plt.suptitle('Time Cost')

        n_t = loc['none']['time']
        z_t = loc['z-score']['time']
        m_t = loc['min-max']['time']
        d_t = loc['decimal']['time']

        times = numpy.array([n_t, z_t, m_t, d_t]) / 3600

        plt.subplot(2, 4, i)
        plt.title(roi + ' - ' + layer)
        plt.ylim(0, 1.5)
        plt.bar(labels, times, color='cornflowerblue')

        for x, y in zip(labels, times):
            plt.text(x, y + 0.05, '%.2f' % y, ha='center', va='bottom')


def dropped_unit_plot(result_data, roi: str, layer: str, mode: str):
    n_d_list = numpy.array([])
    z_d_list = numpy.array([])
    m_d_list = numpy.array([])
    d_d_list = numpy.array([])

    roi_c = roi.upper()

    for i in range(0, 200):
        n_d_list = numpy.append(n_d_list, len(result_data[roi_c][layer]['none']['alpha'][str(i)]))
        z_d_list = numpy.append(z_d_list, len(result_data[roi_c][layer]['z-score']['alpha'][str(i)]))
        m_d_list = numpy.append(m_d_list, len(result_data[roi_c][layer]['min-max']['alpha'][str(i)]))
        d_d_list = numpy.append(d_d_list, len(result_data[roi_c][layer]['decimal']['alpha'][str(i)]))

    if mode == 'gather':
        plt.title('Dropped Units')

        plt.plot(n_d_list, label='No normalization', color='orangered')
        plt.legend()

        plt.plot(d_d_list, label='Decimal Scaling', color='orange')
        plt.legend()

        plt.plot(m_d_list, label='Min-Max', color='lightskyblue')
        plt.legend()

        plt.plot(z_d_list, label='Z-Score', color='cornflowerblue')
        plt.legend()

    elif mode == 'separate':
        plt.suptitle('Dropped Units')

        plt.subplot(411)
        plt.title('No normalization')
        plt.plot(n_d_list, label='No normalization', color='orangered')

        for x, y in zip(range(len(n_d_list)), n_d_list):
            plt.text(x, y + 0.05, y, ha='center', va='bottom')

        plt.subplot(412)
        plt.title('Z-Score')
        plt.plot(z_d_list, label='Z-Score', color='orange')

        for x, y in zip(range(len(z_d_list)), z_d_list):
            plt.text(x, y + 0.05, y, ha='center', va='bottom')

        plt.subplot(413)
        plt.title('Min-Max')
        plt.plot(m_d_list, label='Min-Max', color='lightskyblue')
        plt.legend()

        for x, y in zip(range(len(m_d_list)), m_d_list):
            plt.text(x, y + 0.05, y, ha='center', va='bottom')

        plt.subplot(414)
        plt.title('Decimal Scaling')
        plt.plot(d_d_list, label='Min-Max', color='cornflowerblue')
        plt.legend()

        for x, y in zip(range(len(d_d_list)), d_d_list):
            plt.text(x, y + 0.05, y, ha='center', va='bottom')


def min_max_value_plot(data: dict, roi: str):
    target = data[roi.upper()]
    labels = ['None', 'Z-Score', 'Min-Max', 'Decimal']

    t_min = numpy.array([])
    t_max = numpy.array([])

    y_min = numpy.array([])
    y_max = numpy.array([])

    for t in ['n_train', 'z_train', 'm_train', 'd_train']:
        MIN = min(target[t][0, :])
        MAX = max(target[t][0, :])
        t_min = numpy.append(t_min, MIN)
        t_max = numpy.append(t_max, MAX)

    for F in ['n_y', 'z_y', 'm_y', 'd_y']:
        MIN = min(target[F])
        MAX = max(target[F])

        y_min = numpy.append(y_min, MIN)
        y_max = numpy.append(y_max, MAX)

    bias_max = numpy.abs(t_max - y_max)
    bias_min = -numpy.abs(t_min - y_min)

    plt.suptitle('Min & Max Value')

    plt.subplot(131)
    plt.title('Training data')
    plt.ylim(-10, 10)
    plt.bar(labels, t_max, color='cornflowerblue')
    plt.bar(labels, t_min, color='coral')

    for x, y in zip(labels, t_max):
        plt.text(x, 0.1, '%.3f' % y, ha='center', va='bottom')

    for x, y in zip(labels, t_min):
        plt.text(x, -0.4, '%.3f' % y, ha='center', va='bottom')

    plt.subplot(132)
    plt.title('True data')
    plt.ylim(-10, 10)
    plt.bar(labels, y_max, color='cornflowerblue')
    plt.bar(labels, y_min, color='coral')

    for x, y in zip(labels, y_max):
        plt.text(x, 0.1, '%.5f' % y, ha='center', va='bottom')

    for x, y in zip(labels, y_min):
        plt.text(x, -0.4, '%.5f' % y, ha='center', va='bottom')

    plt.subplot(133)
    plt.title('Bias between Max & Min')
    plt.ylim(-10, 10)
    plt.bar(labels, bias_max, color='cornflowerblue')
    plt.bar(labels, bias_min, color='coral')

    for x, y in zip(labels, bias_max):
        plt.text(x, 0.1, '%.5f' % y, ha='center', va='bottom')

    for x, y in zip(labels, bias_min):
        plt.text(x, -0.4, '%.5f' % y, ha='center', va='bottom')


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


def box_plot(pred_data, layer: str):
    loc = pred_data[layer]['iter_200']
    labels = ['none', 'z-score', 'min-max', 'decimal']

    i = 0
    plt.suptitle('Box Plot - Seen Image Prediction & True')

    for la in labels:
        i += 1

        p_pt = loc[la]['pred_pt'][0, :]
        p_im = loc[la]['pred_im'][0, :]
        t_pt = loc[la]['true_pt'][0, :]
        t_im = loc[la]['true_im'][0, :]

        plt.subplot(2, 2, i)
        plt.title(la)
        plt.boxplot([p_pt, t_pt, p_im, t_im], notch=True, showmeans=True, patch_artist=True,
                    labels=['Seen pred', 'Seen true', 'Imaginary pred', 'Imaginary true'])


def outlier_plot(data: dict, roi: str, mode: str):
    loc = data[roi.upper()]

    labels = ['n_train', 'z_train', 'm_train', 'd_train']
    x_labels = ['None', 'Z-Score', 'Min-Max', 'Decimal']
    colors = ['coral', 'cornflowerblue', 'violet', 'lightgreen']

    if mode == 'bar':
        o_list = numpy.array([])

        for LAB in labels:
            tr = loc[LAB]

            q1 = numpy.percentile(tr, 25)
            q3 = numpy.percentile(tr, 75)
            iqr = q3 - q1
            boundary = iqr * 1.5

            outlier = 0
            for i in range(0, tr.shape[1]):
                num = tr[0, i]
                if num < (q1 - boundary) or num > (q3 + boundary):
                    outlier += 1

            o_list = numpy.append(o_list, outlier)

        plt.title('Number of Outliers')
        plt.bar(x_labels, o_list, color='cornflowerblue')

        for x, y in zip(x_labels, o_list):
            plt.text(x, y + 0.05, y, ha='center', va='bottom')

    elif mode == 'scatter':
        i = 0
        for LAB in labels:
            data = loc[LAB]

            plt.title('Data distribution')
            plt.scatter(data[0, :], data[0, :], color=colors[i], alpha=0.4, label=LAB)
            plt.legend()

            i += 1
    else:
        print('Unknown mode.')


def std_plot(std_s: dict, result_file: dict, roi: str, mode: str):
    loc = std_s[roi.upper()]
    loc_r = result_file[roi.upper()]['cnn1']

    labels = ['none', 'z-score', 'min-max', 'decimal']

    x_test_keys = ['n_test_std', 'z_test_std', 'm_test_std', 'd_test_std']
    y_keys = ['n_y_std', 'z_y_std', 'm_y_std', 'd_y_std']
    x_train_keys = ['n_train_std', 'z_train_std', 'm_train_std', 'd_train_std']

    # Test & Pred ----------------------------------------------------------
    x_test_std = numpy.array([])
    for k in x_test_keys:
        x_test_std = numpy.append(x_test_std, loc[k])

    pred_std = numpy.array([])
    for la in labels:
        pred_data = numpy.append(loc_r[la]['pred_pt'], loc_r[la]['pred_im'])
        pred_std = numpy.append(pred_std, numpy.mean(pred_data))
    # ----------------------------------------------------------------------

    # X & Y --------------------------------------------
    x_train_std = numpy.array([])
    for tr in x_train_keys:
        x_train_std = numpy.append(x_train_std, loc[tr])

    y_std = numpy.array([])
    for k in y_keys:
        y_std = numpy.append(y_std, loc[k])
    # --------------------------------------------------

    plt.suptitle('STD - [%s]' % roi.upper())

    if mode == 'all':
        # Plot 1 ------------------------------------------------------
        plt.subplot(221)
        plt.title('fMRI Data - Test')
        plt.ylim(-5.5, 5.5)
        plt.bar(labels, x_test_std, color='cornflowerblue')

        for x, y in zip(labels, x_test_std):
            plt.text(x, 0.05, '%.5f' % y, ha='center', va='bottom')
        # -------------------------------------------------------------

        # Plot 2 ------------------------------------------------------
        plt.subplot(222)
        plt.title('Prediction')
        plt.ylim(-5.5, 5.5)
        plt.bar(labels, pred_std, color='coral')

        for x, y in zip(labels, pred_std):
            plt.text(x, -0.15, '%.5f' % y, ha='center', va='bottom')
        # -------------------------------------------------------------

        # Plot 3 ------------------------------------------------------
        plt.subplot(223)
        plt.title('fMRI data - Training')
        plt.ylim(-5.5, 5.5)
        plt.bar(labels, x_train_std, color='cornflowerblue')

        for x, y in zip(labels, x_train_std):
            plt.text(x, 0.05, '%.5f' % y, ha='center', va='bottom')
        # -------------------------------------------------------------

        # Plot 4 ------------------------------------------------------
        plt.subplot(224)
        plt.title('Image Feature')
        plt.ylim(-5.5, 5.5)
        plt.bar(labels, y_std, color='coral')

        for x, y in zip(labels, y_std):
            plt.text(x, 0.05, '%.5f' % y, ha='center', va='bottom')
        # -------------------------------------------------------------

    elif mode == 'bias':
        p_bias = x_test_std - pred_std
        t_bias = x_train_std - y_std

        # Plot 1 ------------------------------------------------------
        plt.subplot(121)
        plt.title('Test -> Pred')
        plt.ylim(-5.5, 5.5)
        plt.bar(labels, p_bias, color='cornflowerblue')

        for x, y in zip(labels, p_bias):
            plt.text(x, 0.05, '%.5f' % y, ha='center', va='bottom')
        # -------------------------------------------------------------

        # Plot 2 ------------------------------------------------------
        plt.subplot(122)
        plt.title('Train -> Test')
        plt.ylim(-5.5, 5.5)
        plt.bar(labels, t_bias, color='coral')

        for x, y in zip(labels, t_bias):
            plt.text(x, -0.15, '%.5f' % y, ha='center', va='bottom')
        # -------------------------------------------------------------


def mse_plot(data, roi: str, layer: str):
    norm = ['none', 'z-score', 'min-max', 'decimal']

    i = 0
    plt.suptitle('MSE - [%s, %s]' % (roi.upper(), layer))

    for n in norm:
        i += 1
        pred_pt = data[roi.upper()][layer][n]['pred_pt']
        true_pt = data[roi.upper()][layer][n]['true_pt']

        pred_im = data[roi.upper()][layer][n]['pred_im']
        true_im = data[roi.upper()][layer][n]['true_im']

        pt_mse = sklearn.mean_squared_error(true_pt, pred_pt)
        im_mse = sklearn.mean_squared_error(true_im, pred_im)

        print('pt_mse: ', pt_mse)
        print('im_mse: ', im_mse)

        mse_list = numpy.array([pt_mse, im_mse])

        plt.subplot(1, 4, i)
        plt.title(n)
        plt.bar(['Seen', 'Imaginary'], mse_list, color=['cornflowerblue', 'coral'])

        for x, y in zip(['Seen', 'Imaginary'], mse_list):
            plt.text(x, y + 0.05, '%.5f' % y, ha='center', va='bottom')


def var_pred_plot(data, roi: str, layer: str):
    roi_up = roi.upper()
    norm = ['none', 'z-score', 'min-max', 'decimal']

    plt.suptitle('Variance - [V1, cnn1] ')

    x_label = ['None', 'Z-Score', 'Min-Max', 'Decimal', 'True']

    n_pred_pt = data[roi_up][layer][norm[0]]['pred_pt']
    z_pred_pt = data[roi_up][layer][norm[1]]['pred_pt']
    m_pred_pt = data[roi_up][layer][norm[2]]['pred_pt']
    d_pred_pt = data[roi_up][layer][norm[3]]['pred_pt']

    n_pt_v = numpy.sum(numpy.power(n_pred_pt - numpy.mean(n_pred_pt), 2)) / (1750 * 1000)
    z_pt_v = numpy.sum(numpy.power(z_pred_pt - numpy.mean(z_pred_pt), 2)) / (1750 * 1000)
    m_pt_v = numpy.sum(numpy.power(m_pred_pt - numpy.mean(m_pred_pt), 2)) / (1750 * 1000)
    d_pt_v = numpy.sum(numpy.power(d_pred_pt - numpy.mean(d_pred_pt), 2)) / (1750 * 1000)

    n_pred_im = data[roi_up][layer][norm[0]]['pred_im']
    z_pred_im = data[roi_up][layer][norm[1]]['pred_im']
    m_pred_im = data[roi_up][layer][norm[2]]['pred_im']
    d_pred_im = data[roi_up][layer][norm[3]]['pred_im']

    n_im_v = numpy.sum(numpy.power(n_pred_im - numpy.mean(n_pred_im), 2)) / (500 * 1000)
    z_im_v = numpy.sum(numpy.power(z_pred_im - numpy.mean(z_pred_im), 2)) / (500 * 1000)
    m_im_v = numpy.sum(numpy.power(m_pred_im - numpy.mean(m_pred_im), 2)) / (500 * 1000)
    d_im_v = numpy.sum(numpy.power(d_pred_im - numpy.mean(d_pred_im), 2)) / (500 * 1000)

    true_pt = data[roi_up][layer][norm[0]]['true_pt']
    true_im = data[roi_up][layer][norm[0]]['true_im']

    t_pt_v = numpy.sum(numpy.power(true_pt - numpy.mean(true_pt), 2)) / (1750 * 1000)
    t_im_v = numpy.sum(numpy.power(true_im - numpy.mean(true_im), 2)) / (500 * 1000)

    p_v_list = [n_pt_v, z_pt_v, m_pt_v, d_pt_v, t_pt_v]
    i_v_list = [n_im_v, z_im_v, m_im_v, d_im_v, t_im_v]

    plt.subplot(121)
    plt.title('PT')
    plt.bar(x_label, p_v_list, color='cornflowerblue')

    for x, y in zip(x_label, p_v_list):
        plt.text(x, y + 0.5, '%.5f' % y, ha='center', va='bottom')

    plt.subplot(122)
    plt.title('IM')
    plt.bar(x_label, i_v_list, color='coral')

    for x, y in zip(x_label, i_v_list):
        plt.text(x, y + 0.5, '%.5f' % y, ha='center', va='bottom')


def var_train_plot(data, roi: str):
    roi_up = roi.upper()
    norm = ['none', 'z-score', 'min-max', 'decimal']

    loc = data[roi_up]

    n_train = loc['n_train']
    z_train = loc['z_train']
    m_train = loc['m_train']
    d_train = loc['d_train']

    n_v = numpy.sum(numpy.power(n_train - numpy.mean(n_train), 2)) / 600000
    z_v = numpy.sum(numpy.power(z_train - numpy.mean(z_train), 2)) / 600000
    m_v = numpy.sum(numpy.power(m_train - numpy.mean(m_train), 2)) / 600000
    d_v = numpy.sum(numpy.power(d_train - numpy.mean(d_train), 2)) / 600000

    n_y = loc['n_y']
    z_y = loc['z_y']
    m_y = loc['m_y']
    d_y = loc['d_y']

    n_y_v = numpy.sum(numpy.power(n_y - numpy.mean(n_y), 2)) / 1200
    z_y_v = numpy.sum(numpy.power(z_y - numpy.mean(z_y), 2)) / 1200
    m_y_v = numpy.sum(numpy.power(m_y - numpy.mean(m_y), 2)) / 1200
    d_y_v = numpy.sum(numpy.power(d_y - numpy.mean(d_y), 2)) / 1200

    x_list = [n_v, z_v, m_v, d_v]
    y_list = [n_y_v, z_y_v, m_y_v, d_y_v]
    bias_list = [n_v - n_y_v, z_v - z_y_v, m_v - m_y_v, d_v - d_y_v]

    plt.suptitle('Variance of Training data - [%s]' % roi_up)

    plt.subplot(131)
    plt.title('X')
    plt.bar(norm, x_list, color='cornflowerblue')
    for x, y in zip(norm, x_list):
        plt.text(x, y + 0.05, '%.05f' % y, ha='center', va='bottom')

    plt.subplot(132)
    plt.title('Y')
    plt.bar(norm, y_list, color='coral')
    for x, y in zip(norm, y_list):
        plt.text(x, y + 0.01, '%.05f' % y, ha='center', va='bottom')

    plt.subplot(133)
    plt.title('Trend')
    plt.bar(norm, bias_list, color='forestgreen')
    for x, y in zip(norm, bias_list):
        plt.text(x, y + 0.05, '%.05f' % y, ha='center', va='bottom')


# ---------------------------------------------------
def norm_trainSet_contrast(title: str, pattern: int):
    # Compare original data and after normalization
    TOOL = Tools.Tool()
    # Pattern:
    # 0 = get bias for each image feature
    # 1 = get the min & max value
    # 2 = Negative value occupation

    # Get x ---------------------------------
    data, image_feature = TOOL.read_subject_1()
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
def cor_merged_av_plot(data: dict, roi: str, layer: str):
    loc = data[roi.upper()][layer]

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

        mean_set = [mean_pt, mean_im]
        mean_cat_set = [mean_cat_pt, mean_cat_im]

        x_label_tru = ['mean_pt', 'mean_im']
        x_label_cat = ['mean_cat_pt', 'mean_cat_im']

        y_lim = 0.13

        plt.subplot(2, 4, i)
        plt.title(n_key + ' - ' + '[Pred - True]')
        plt.ylim(0, y_lim)
        plt.bar(x_label_tru, mean_set, color=['cornflowerblue', 'coral'])
        plt.legend()

        for x, y in zip(x_label_tru, mean_set):
            plt.text(x, y + 0.0005, '%.5f' % y, ha='center', va='bottom')

        plt.subplot(2, 4, i + 4)
        plt.title(n_key + ' - ' + '[Pred - Categories]')
        plt.ylim(0, y_lim)
        plt.bar(x_label_cat, mean_cat_set, color=['cornflowerblue', 'coral'])
        plt.legend()

        for x, y in zip(x_label_cat, mean_cat_set):
            plt.text(x, y + 0.0005, '%.5f' % y, ha='center', va='bottom')


def cor_av_separate_plot(data: dict, roi: str, layer: str):
    loc = data[roi.upper()][layer]
    d_keys = ['cor_pt_av', 'cor_im_av', 'cor_cat_pt_av', 'cor_cat_im_av']
    norm = ['None[ PT ]', 'Z-Score[ PT ]', 'Min-Max[ PT ]', 'Decimal[ PT ]',
            'None[ IM ]', 'Z-Score[ IM ]', 'Min-Max[ IM ]', 'Decimal[ IM ]']
    target = ['- [Pred - True]', '- [Pred - CAT]']

    plt.suptitle('Correlation for Merged Result')

    pt_n_cor_av = loc['none'][d_keys[0]].reshape(50)
    pt_z_cor_av = loc['z-score'][d_keys[0]].reshape(50)
    pt_m_cor_av = loc['min-max'][d_keys[0]].reshape(50)
    pt_d_cor_av = loc['decimal'][d_keys[0]].reshape(50)

    pt_n_cor_cat_av = loc['none'][d_keys[2]].reshape(50)
    pt_z_cor_cat_av = loc['z-score'][d_keys[2]].reshape(50)
    pt_m_cor_cat_av = loc['min-max'][d_keys[2]].reshape(50)
    pt_d_cor_cat_av = loc['decimal'][d_keys[2]].reshape(50)

    im_n_cor_av = loc['none'][d_keys[1]].reshape(50)
    im_z_cor_av = loc['z-score'][d_keys[1]].reshape(50)
    im_m_cor_av = loc['min-max'][d_keys[1]].reshape(50)
    im_d_cor_av = loc['decimal'][d_keys[1]].reshape(50)
    im_n_cor_cat_av = loc['none'][d_keys[3]].reshape(50)
    im_z_cor_cat_av = loc['z-score'][d_keys[3]].reshape(50)
    im_m_cor_cat_av = loc['min-max'][d_keys[3]].reshape(50)
    im_d_cor_cat_av = loc['decimal'][d_keys[3]].reshape(50)

    all_data = {'%s %s' % (norm[0], target[0]): pt_n_cor_av, '%s %s' % (norm[1], target[0]): pt_z_cor_av,
                '%s %s' % (norm[2], target[0]): pt_m_cor_av, '%s %s' % (norm[3], target[0]): pt_d_cor_av,
                '%s %s' % (norm[0], target[1]): pt_n_cor_cat_av, '%s %s' % (norm[1], target[1]): pt_z_cor_cat_av,
                '%s %s' % (norm[2], target[1]): pt_m_cor_cat_av, '%s %s' % (norm[3], target[1]): pt_d_cor_cat_av,
                '%s %s' % (norm[4], target[0]): im_n_cor_av, '%s %s' % (norm[5], target[0]): im_z_cor_av,
                '%s %s' % (norm[6], target[0]): im_m_cor_av, '%s %s' % (norm[7], target[0]): im_d_cor_av,
                '%s %s' % (norm[4], target[1]): im_n_cor_cat_av, '%s %s' % (norm[5], target[1]): im_z_cor_cat_av,
                '%s %s' % (norm[6], target[1]): im_m_cor_cat_av, '%s %s' % (norm[7], target[1]): im_d_cor_cat_av,
                }

    plt.suptitle('Correlation Coefficient')

    i = 0
    for key in all_data:
        i += 1
        plt.subplot(4, 4, i)
        plt.xlabel('Correlation')
        plt.ylabel('Images - [ALL=50]')
        plt.title(key)
        plt.hist(all_data[key], alpha=0.3)


# =====================================================================================
# -------------------------------------------------------------------------------------
tool = Tools.Tool()

result = tool.read_result_data('s1')
merged_result = tool.read_merged_data('s1')
xy_train_std = tool.read_xy_std_data('s1')
