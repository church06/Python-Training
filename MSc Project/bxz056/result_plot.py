import bdpy
import numpy
import sklearn.metrics as sklearn
from bdpy.preproc import select_top
from bdpy.stats import corrcoef
from matplotlib import pyplot as plt

import Tools


def time_plot(data: dict):
    roi_s = list(data.keys())

    plt.figure(figsize=(30, 15))
    plt.suptitle('Time Cost')

    i = 0
    for R in roi_s:
        for L in ['cnn1', 'cnn2', 'cnn4', 'cnn6', 'cnn8', 'hmax1', 'hmax2', 'hmax3', 'gist', 'sift']:
            i += 1
            loc = data[R][L]
            labels = ['N', 'Z', 'M', 'D']

            plt.suptitle('Time Cost')

            n_t = loc['none']['time']
            z_t = loc['z-score']['time']
            m_t = loc['min-max']['time']
            d_t = loc['decimal']['time']

            times = numpy.array([n_t, z_t, m_t, d_t]) / 3600

            plt.subplot(7, 10, i)
            plt.title(R + ' - ' + L)
            plt.ylim(0, 1.5)
            plt.bar(labels, times, color='cornflowerblue')
            plt.grid(True)

            for x, y in zip(labels, times):
                plt.text(x, y + 0.05, '%.2f' % y, ha='center', va='bottom')

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig('plots\\results\\time_cost.png')
    plt.close('all')


def dropped_unit_plot(result_data, roi: str, mode: str):
    cnn = ['cnn1', 'cnn2', 'cnn4', 'cnn6', 'cnn8']
    hmax = ['hmax1', 'hmax2', 'hmax3']
    others = ['gist', 'sift']

    roi_c = roi.upper()
    i = 0
    plt.suptitle('Dropped Units - [%s]' % roi)

    if mode == 'cnn':
        for C in cnn:
            n_d_list = numpy.array([])
            z_d_list = numpy.array([])
            m_d_list = numpy.array([])
            d_d_list = numpy.array([])
            i += 1

            for index in range(0, 200):
                n_d_list = numpy.append(n_d_list, len(result_data[roi_c][C]['none']['alpha'][str(index)]))
                z_d_list = numpy.append(z_d_list, len(result_data[roi_c][C]['z-score']['alpha'][str(index)]))
                m_d_list = numpy.append(m_d_list, len(result_data[roi_c][C]['min-max']['alpha'][str(index)]))
                d_d_list = numpy.append(d_d_list, len(result_data[roi_c][C]['decimal']['alpha'][str(index)]))

            plt.subplot(3, 2, i)
            plt.title(C)
            plt.plot(n_d_list, label='No normalization', color='orangered')

            plt.subplot(3, 2, i)
            plt.title(C)
            plt.plot(d_d_list, label='Decimal Scaling', color='orange')

            plt.subplot(3, 2, i)
            plt.title(C)
            plt.plot(m_d_list, label='Min-Max', color='lightskyblue')

            plt.subplot(3, 2, i)
            plt.title(C)
            plt.plot(z_d_list, label='Z-Score', color='cornflowerblue')
            plt.legend(['N', 'Z', 'M', 'D'])

    elif mode == 'hmax':
        for H in hmax:
            n_d_list = numpy.array([])
            z_d_list = numpy.array([])
            m_d_list = numpy.array([])
            d_d_list = numpy.array([])
            i += 1

            for index in range(0, 200):
                n_d_list = numpy.append(n_d_list, len(result_data[roi_c][H]['none']['alpha'][str(index)]))
                z_d_list = numpy.append(z_d_list, len(result_data[roi_c][H]['z-score']['alpha'][str(index)]))
                m_d_list = numpy.append(m_d_list, len(result_data[roi_c][H]['min-max']['alpha'][str(index)]))
                d_d_list = numpy.append(d_d_list, len(result_data[roi_c][H]['decimal']['alpha'][str(index)]))

            plt.subplot(1, 3, i)
            plt.title(H)
            plt.plot(n_d_list, label='No normalization', color='orangered')

            plt.subplot(1, 3, i)
            plt.title(H)
            plt.plot(d_d_list, label='Decimal Scaling', color='orange')

            plt.subplot(1, 3, i)
            plt.title(H)
            plt.plot(m_d_list, label='Min-Max', color='lightskyblue')

            plt.subplot(1, 3, i)
            plt.title(H)
            plt.plot(z_d_list, label='Z-Score', color='cornflowerblue')
            plt.legend(['N', 'Z', 'M', 'D'])

    elif mode == 'others':
        for Ot in others:
            n_d_list = numpy.array([])
            z_d_list = numpy.array([])
            m_d_list = numpy.array([])
            d_d_list = numpy.array([])
            i += 1

            for index in range(0, 200):
                n_d_list = numpy.append(n_d_list, len(result_data[roi_c][Ot]['none']['alpha'][str(index)]))
                z_d_list = numpy.append(z_d_list, len(result_data[roi_c][Ot]['z-score']['alpha'][str(index)]))
                m_d_list = numpy.append(m_d_list, len(result_data[roi_c][Ot]['min-max']['alpha'][str(index)]))
                d_d_list = numpy.append(d_d_list, len(result_data[roi_c][Ot]['decimal']['alpha'][str(index)]))

            plt.subplot(1, 2, i)
            plt.title(Ot)
            plt.plot(n_d_list, label='No normalization', color='orangered')

            plt.subplot(1, 2, i)
            plt.title(Ot)
            plt.plot(d_d_list, label='Decimal Scaling', color='orange')

            plt.subplot(1, 2, i)
            plt.title(Ot)
            plt.plot(m_d_list, label='Min-Max', color='lightskyblue')

            plt.subplot(1, 2, i)
            plt.title(Ot)
            plt.plot(z_d_list, label='Z-Score', color='cornflowerblue')
            plt.legend(['N', 'Z', 'M', 'D'])
    else:
        print('Unknown mode.')
        return None


def min_max_value_plot(data: dict, roi: str, mode: str):
    target = data[roi.upper()]

    cnn = ['cnn1', 'cnn2', 'cnn4', 'cnn6', 'cnn8']
    hmax = ['hmax1', 'hmax2', 'hmax3']
    others = ['gist', 'sift']

    if mode == 'cnn':
        min_max_draw(target, cnn, 3, 6, roi)

    elif mode == 'hmax':
        min_max_draw(target, hmax, 3, 3, roi)

    elif mode == 'others':
        min_max_draw(target, others, 2, 3, roi)


def min_max_draw(data: dict, layers: list, ax_x: int, ax_y: int, roi: str):
    labels = ['None', 'Z-Score', 'Min-Max', 'Decimal']
    i = 0

    for L in layers:
        t_min, t_max = [], []
        y_min, y_max = [], []

        i += 1

        for T in ['n_train', 'z_train', 'm_train', 'd_train']:
            MIN = min(data[L][T][0, :])
            MAX = max(data[L][T][0, :])
            t_min = numpy.append(t_min, MIN)
            t_max = numpy.append(t_max, MAX)

        for F in ['n_y', 'z_y', 'm_y', 'd_y']:
            MIN = min(data['cnn1'][F])
            MAX = max(data['cnn1'][F])

            y_min = numpy.append(y_min, MIN)
            y_max = numpy.append(y_max, MAX)

        bias_max = numpy.abs(t_max - y_max)
        bias_min = -numpy.abs(t_min - y_min)

        plt.suptitle('Min & Max Value - %s' % roi)

        plt.subplot(ax_x, ax_y, i)
        plt.title('Training data - %s' % L)
        plt.ylim(-10, 10)
        plt.bar(labels, t_max, color='cornflowerblue')
        plt.bar(labels, t_min, color='coral')

        i += 1

        plt.subplot(ax_x, ax_y, i)
        plt.title('True data - %s' % L)
        plt.ylim(-10, 10)
        plt.bar(labels, y_max, color='cornflowerblue')
        plt.bar(labels, y_min, color='coral')

        i += 1

        plt.subplot(ax_x, ax_y, i)
        plt.title('Bias between Max & Min - %s' % L)
        plt.ylim(-10, 10)
        plt.bar(labels, bias_max, color='cornflowerblue')
        plt.bar(labels, bias_min, color='coral')

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig('plots\\results\\min_max_%s_%s.png' % (roi, layers[0][:-1]))
    plt.close('all')


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
        plt.figure(figsize=(20, 11))
        plt.suptitle('Outliers - [%s]' % roi)
        index = 0

        for L in ['cnn1', 'cnn2', 'cnn4', 'cnn6', 'cnn8', 'hmax1', 'hmax2', 'hmax3', 'gist', 'sift']:
            index += 1
            o_list = numpy.array([])
            for LAB in labels:
                tr = loc[L][LAB]

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

            plt.subplot(2, 5, index)
            plt.title(L)
            plt.bar(x_labels, o_list, color='cornflowerblue')

            for x, y in zip(x_labels, o_list):
                plt.text(x, y + 0.05, y, ha='center', va='bottom')

        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig('plots\\results\\outliers_%s.png' % roi)
        plt.close('all')

    elif mode == 'scatter':
        index = 0

        plt.figure(figsize=(20, 60))
        plt.suptitle('Data Distribution')

        for R in ['v1', 'v2', 'v3', 'v4', 'loc', 'ffa', 'ppa']:
            file = data[R.upper()]

            for L in ['cnn1', 'cnn2', 'cnn4', 'cnn6', 'cnn8', 'hmax1', 'hmax2', 'hmax3', 'gist', 'sift']:
                index += 1
                i = 0

                for LAB in labels:
                    target = file[L][LAB]

                    plt.subplot(14, 5, index)
                    plt.title('%s - [%s]' % (R, L))
                    plt.ylim(-30, 30)
                    plt.xlim(-30, 30)
                    plt.scatter(target[0, :], target[0, :], color=colors[i], alpha=0.4, label=LAB)
                    plt.legend()

                    i += 1

        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.savefig('plots\\results\\distribution.png')
        plt.close('all')

    else:
        print('Unknown mode.')


def std_plot(std_s: dict):
    i = 0
    plt.figure(figsize=(40, 100))
    for R in ['v1', 'v2', 'v3', 'v4', 'loc', 'ffa', 'ppa']:
        for L in ['cnn1', 'cnn2', 'cnn4', 'cnn6', 'cnn8', 'hmax1', 'hmax2', 'hmax3', 'gist', 'sift']:
            i += 1
            loc = std_s[R.upper()][L]

            labels = ['none', 'z-score', 'min-max', 'decimal']

            y_keys = ['n_y_std', 'z_y_std', 'm_y_std', 'd_y_std']
            x_train_keys = ['n_train_std', 'z_train_std', 'm_train_std', 'd_train_std']

            # X & Y --------------------------------------------
            x_train_std = numpy.array([])
            for tr in x_train_keys:
                x_train_std = numpy.append(x_train_std, loc[tr])

            y_std = numpy.array([])
            for k in y_keys:
                y_std = numpy.append(y_std, loc[k])
            # --------------------------------------------------

            plt.suptitle('Standard Deviation')

            # Plot 1 --------------------------------------------------
            plt.subplot(35, 4, i)
            plt.title('fMRI data - [%s %s]' % (R, L))
            plt.ylim(-5.5, 5.5)
            plt.bar(labels, x_train_std, color='cornflowerblue')

            for x, y in zip(labels, x_train_std):
                plt.text(x, 0.05, '%.5f' % y, ha='center', va='bottom')
            # ---------------------------------------------------------

            i += 1

            # Plot 2 ------------------------------------------------------
            plt.subplot(35, 4, i)
            plt.title('Image Feature - [%s %s]' % (R, L))
            plt.ylim(-5.5, 5.5)
            plt.bar(labels, y_std, color='coral')

            for x, y in zip(labels, y_std):
                plt.text(x, 0.05, '%.5f' % y, ha='center', va='bottom')
            # -------------------------------------------------------------

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig('plots\\results\\std.png')
    plt.close('all')


def var_pred_plot(data, roi: str):
    roi_up = roi.upper()
    norm_s = ['none', 'z-score', 'min-max', 'decimal']
    x_label = ['None', 'Z-Score', 'Min-Max', 'Decimal', 'True']

    plt.suptitle('Variance - [%s]' % roi)

    i = 0
    for L in ['cnn1', 'cnn2', 'cnn4', 'cnn6', 'cnn8', 'hmax1', 'hmax2', 'hmax3', 'gist', 'sift']:
        i += 1

        n_pred_pt = data[roi_up][L][norm_s[0]]['pred_pt']
        z_pred_pt = data[roi_up][L][norm_s[1]]['pred_pt']
        m_pred_pt = data[roi_up][L][norm_s[2]]['pred_pt']
        d_pred_pt = data[roi_up][L][norm_s[3]]['pred_pt']

        n_pt_v = numpy.sum(numpy.power(n_pred_pt - numpy.mean(n_pred_pt), 2)) / (1750 * 1000)
        z_pt_v = numpy.sum(numpy.power(z_pred_pt - numpy.mean(z_pred_pt), 2)) / (1750 * 1000)
        m_pt_v = numpy.sum(numpy.power(m_pred_pt - numpy.mean(m_pred_pt), 2)) / (1750 * 1000)
        d_pt_v = numpy.sum(numpy.power(d_pred_pt - numpy.mean(d_pred_pt), 2)) / (1750 * 1000)

        n_pred_im = data[roi_up][L][norm_s[0]]['pred_im']
        z_pred_im = data[roi_up][L][norm_s[1]]['pred_im']
        m_pred_im = data[roi_up][L][norm_s[2]]['pred_im']
        d_pred_im = data[roi_up][L][norm_s[3]]['pred_im']

        n_im_v = numpy.sum(numpy.power(n_pred_im - numpy.mean(n_pred_im), 2)) / (500 * 1000)
        z_im_v = numpy.sum(numpy.power(z_pred_im - numpy.mean(z_pred_im), 2)) / (500 * 1000)
        m_im_v = numpy.sum(numpy.power(m_pred_im - numpy.mean(m_pred_im), 2)) / (500 * 1000)
        d_im_v = numpy.sum(numpy.power(d_pred_im - numpy.mean(d_pred_im), 2)) / (500 * 1000)

        true_pt = data[roi_up][L][norm_s[0]]['true_pt']
        true_im = data[roi_up][L][norm_s[0]]['true_im']

        t_pt_v = numpy.sum(numpy.power(true_pt - numpy.mean(true_pt), 2)) / (1750 * 1000)
        t_im_v = numpy.sum(numpy.power(true_im - numpy.mean(true_im), 2)) / (500 * 1000)

        p_v_list = [n_pt_v, z_pt_v, m_pt_v, d_pt_v, t_pt_v]
        i_v_list = [n_im_v, z_im_v, m_im_v, d_im_v, t_im_v]

        plt.subplot(5, 4, i)
        plt.title('%s - PT' % L)
        plt.bar(x_label, p_v_list, color='cornflowerblue')

        i += 1

        plt.subplot(5, 4, i)
        plt.title('%s - IM' % L)
        plt.bar(x_label, i_v_list, color='coral')


def var_train_plot(data, roi: str):
    roi_up = roi.upper()
    norm_s = ['none', 'z-score', 'min-max', 'decimal']

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
    plt.bar(norm_s, x_list, color='cornflowerblue')
    for x, y in zip(norm_s, x_list):
        plt.text(x, y + 0.05, '%.05f' % y, ha='center', va='bottom')

    plt.subplot(132)
    plt.title('Y')
    plt.bar(norm_s, y_list, color='coral')
    for x, y in zip(norm_s, y_list):
        plt.text(x, y + 0.01, '%.05f' % y, ha='center', va='bottom')

    plt.subplot(133)
    plt.title('Trend')
    plt.bar(norm_s, bias_list, color='forestgreen')
    for x, y in zip(norm_s, bias_list):
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
def cor_av_plot(data: dict, roi: list, layer: str, mode: str):
    d_keys = ['cor_pt_av', 'cor_im_av']
    plt.suptitle('Correlation Coefficient - [%s %s]' % (layer, mode.upper()))

    cor_all = {}
    for R in roi:
        loc = data[R][layer]
        cor_r = {}

        if mode == 'pt':
            n_cor = loc['none'][d_keys[0]].reshape(50)
            z_cor = loc['z-score'][d_keys[0]].reshape(50)
            m_cor = loc['min-max'][d_keys[0]].reshape(50)
            d_cor = loc['decimal'][d_keys[0]].reshape(50)

            cor_r = {'None': n_cor, 'Z-Score': z_cor, 'Min-MAX': m_cor, 'Decimal Scaling': d_cor}
        elif mode == 'im':
            n_cor = loc['none'][d_keys[1]].reshape(50)
            z_cor = loc['z-score'][d_keys[1]].reshape(50)
            m_cor = loc['min-max'][d_keys[1]].reshape(50)
            d_cor = loc['decimal'][d_keys[1]].reshape(50)

            cor_r = {'None': n_cor, 'Z-Score': z_cor, 'Min-MAX': m_cor, 'Decimal Scaling': d_cor}

        cor_all[R] = cor_r

    i = 0
    for roi in cor_all:
        i += 1
        out_list = []
        for k in cor_all[roi]:
            out_list.append(cor_all[roi][k])

        plt.subplot(4, 2, i)
        plt.title(roi)
        plt.hist(out_list, alpha=0.4)
        plt.legend(['None', 'Z-Score', 'Min-Max', 'Decimal Scaling'], loc='best')
        plt.grid(True)


def ac_separate_plot(data: dict, roi: list, layer: str):
    d_keys = ['n_cr_pt', 'z_cr_pt', 'm_cr_pt', 'd_cr_pt', 'n_cr_im', 'z_cr_im', 'm_cr_im', 'd_cr_im']
    labels = ['None - [ PT ]', 'Z-Score - [ PT ]', 'Min-Max - [ PT ]', 'Decimal - [ PT ]',
              'None - [ IM ]', 'Z-Score - [ IM ]', 'Min-Max - [ IM ]', 'Decimal - [ IM ]']

    v1 = {}
    v2 = {}
    v3 = {}
    v4 = {}
    loc = {}
    ffa = {}
    ppa = {}

    plt.figure(figsize=(25, 15))
    plt.suptitle('Accuracy - [ %s ]' % layer.upper())

    for R in roi:
        target = data[R.upper()][layer]

        n_pt = target[d_keys[0]].reshape(50)
        z_pt = target[d_keys[1]].reshape(50)
        m_pt = target[d_keys[2]].reshape(50)
        d_pt = target[d_keys[3]].reshape(50)

        n_im = target[d_keys[4]].reshape(50)
        z_im = target[d_keys[5]].reshape(50)
        m_im = target[d_keys[6]].reshape(50)
        d_im = target[d_keys[7]].reshape(50)

        if R == 'v1':
            v1 = {labels[0]: n_pt, labels[1]: z_pt, labels[2]: m_pt, labels[3]: d_pt,
                  labels[4]: n_im, labels[5]: z_im, labels[6]: m_im, labels[7]: d_im}

        elif R == 'v2':
            v2 = {labels[0]: n_pt, labels[1]: z_pt, labels[2]: m_pt, labels[3]: d_pt,
                  labels[4]: n_im, labels[5]: z_im, labels[6]: m_im, labels[7]: d_im}

        elif R == 'v3':
            v3 = {labels[0]: n_pt, labels[1]: z_pt, labels[2]: m_pt, labels[3]: d_pt,
                  labels[4]: n_im, labels[5]: z_im, labels[6]: m_im, labels[7]: d_im}

        elif R == 'v4':
            v4 = {labels[0]: n_pt, labels[1]: z_pt, labels[2]: m_pt, labels[3]: d_pt,
                  labels[4]: n_im, labels[5]: z_im, labels[6]: m_im, labels[7]: d_im}

        elif R == 'loc':
            loc = {labels[0]: n_pt, labels[1]: z_pt, labels[2]: m_pt, labels[3]: d_pt,
                   labels[4]: n_im, labels[5]: z_im, labels[6]: m_im, labels[7]: d_im}

        elif R == 'ffa':
            ffa = {labels[0]: n_pt, labels[1]: z_pt, labels[2]: m_pt, labels[3]: d_pt,
                   labels[4]: n_im, labels[5]: z_im, labels[6]: m_im, labels[7]: d_im}

        elif R == 'ppa':
            ppa = {labels[0]: n_pt, labels[1]: z_pt, labels[2]: m_pt, labels[3]: d_pt,
                   labels[4]: n_im, labels[5]: z_im, labels[6]: m_im, labels[7]: d_im}

    all_data = {'v1': v1, 'v2': v2, 'v3': v3, 'v4': v4, 'loc': loc, 'ffa': ffa, 'ppa': ppa}

    i = 0
    for key in v1:
        used_data = []

        for R in roi:
            used_data.append(all_data[R][key])

        i += 1
        plt.subplot(2, 4, i)

        plt.xlabel('Accuracy')
        plt.ylabel('Images - [ALL=50]')
        plt.xlim(0, 1.1)
        plt.title(key)

        plt.hist(used_data, alpha=0.5)
        plt.grid(True)
        plt.legend(roi)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig('plots\\results\\ac_%s_all.png' % layer)
    plt.close('all')


def ac_average_plot(data: dict, mode: str):
    roi = ['V1', 'V2', 'V3', 'V4', 'LOC', 'FFA', 'PPA']
    layers = ['cnn1', 'cnn2', 'cnn4', 'cnn6', 'cnn8',
              'hmax1', 'hmax2', 'hmax3',
              'gist', 'sift']

    i = 0
    plt.figure(figsize=(50, 30))
    plt.suptitle('Accuracy Average')

    for R in roi:
        i += 1
        loc = data[R]

        n_all = []
        z_all = []
        m_all = []
        d_all = []

        for L in layers:
            if mode == 'pt':
                n_cr = loc[L]['n_cr_pt_av']
                z_cr = loc[L]['z_cr_pt_av']
                m_cr = loc[L]['m_cr_pt_av']
                d_cr = loc[L]['d_cr_pt_av']

                n_all.append(n_cr)
                z_all.append(z_cr)
                m_all.append(m_cr)
                d_all.append(d_cr)

            elif mode == 'im':
                n_cr = loc[L]['n_cr_im_av']
                z_cr = loc[L]['z_cr_im_av']
                m_cr = loc[L]['m_cr_im_av']
                d_cr = loc[L]['d_cr_im_av']

                n_all.append(n_cr)
                z_all.append(z_cr)
                m_all.append(m_cr)
                d_all.append(d_cr)

            else:
                print('Unknown mode.')
                return None

        none = range(0, 100, 10)
        z_s = range(120, 220, 10)
        m_m = range(240, 340, 10)
        d_s = range(360, 460, 10)

        plt.subplot(4, 2, i)
        plt.title(R + ' - [%s]' % mode.upper())

        plt.bar(none, n_all, alpha=0.4, width=8.5)
        plt.ylim(0, 1)
        plt.xticks(none, layers)

        plt.bar(z_s, n_all, alpha=0.4, width=8.5)
        plt.ylim(0, 1)

        plt.bar(m_m, n_all, alpha=0.4, width=8.5)
        plt.ylim(0, 1)

        plt.bar(d_s, n_all, alpha=0.4, width=8.5)
        plt.ylim(0, 1)

        plt.legend(['None', 'Z-Score', 'Min-Max', 'Decimal Scaling'], loc='best')
        plt.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig('plots\\results\\ac_av_%s.png' % mode)
    plt.close('all')


def cor_cat_plot(data: dict, mode: str):
    plt.figure(figsize=(30, 15))
    plt.suptitle('Correlation Coefficient [Pred - CAT]')

    i = 0
    for L in ['cnn1', 'cnn2', 'cnn4', 'cnn6', 'cnn8']:
        for R in list(data.keys()):
            i += 1
            cr_list = []

            if mode == 'pt':
                n_cr = data[R][L]['n_cr_pt_av']
                z_cr = data[R][L]['z_cr_pt_av']
                m_cr = data[R][L]['m_cr_pt_av']
                d_cr = data[R][L]['d_cr_pt_av']

                cr_list = [n_cr, z_cr, m_cr, d_cr]

            elif mode == 'im':
                n_cr = data[R][L]['n_cr_im_av']
                z_cr = data[R][L]['z_cr_im_av']
                m_cr = data[R][L]['m_cr_im_av']
                d_cr = data[R][L]['d_cr_im_av']

                cr_list = [n_cr, z_cr, m_cr, d_cr]

            plt.subplot(5, 7, i)
            plt.title(L + ' - ' + R + ' [ %s ]' % mode.upper())
            plt.bar(['none', 'z-score', 'min-max', 'decimal'], cr_list)
            plt.ylim(0, 1.1)
            plt.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig('plots\\results\\cor_cat_%s.png' % mode)
    plt.close('all')


def gaussian_distribution_plot(data: dict, roi: str, layer: str):
    cnn = ['cnn1', 'cnn2', 'cnn3', 'cnn4', 'cnn5', 'cnn6', 'cnn7', 'cnn8']
    hmax = ['hmax1', 'hmax2', 'hmax3']
    others = ['gist', 'sift']

    label = ['None', 'Z-Score', 'Min-Max', 'Decimal Scaling']

    roi_up = roi.upper()
    i = 1

    if layer == 'cnn':
        plt.figure(figsize=(20, 40))

        for C in cnn:
            loc = data[roi_up][C]

            for N in label:
                term = N[0].lower()

                x = loc[term + '_train']
                y = loc[term + '_y']

                plt.subplot(8, 2, i)
                plt.title('fMRI [%s - %s]' % (roi, C))
                plt.hist(x, alpha=0.3)
                plt.xlim(-80, 80)
                plt.legend(label)
                plt.grid(True)

                plt.subplot(8, 2, i + 1)
                plt.title('Image Feature [%s - %s]' % (roi, C))
                plt.hist(y, alpha=0.3)
                plt.legend(label)
                plt.grid(True)

            i += 2

        plt.tight_layout()
        plt.savefig('plots\\results\\train_hist_%s.png' % layer)
        plt.close('all')

    elif layer == 'hmax':
        plt.figure(figsize=(20, 15))

        for H in hmax:
            loc = data[roi_up][H]

            for N in label:
                term = N[0].lower()

                x = loc[term + '_train']
                y = loc[term + '_y']

                plt.subplot(8, 2, i)
                plt.title('fMRI [%s - %s]' % (roi, layer))
                plt.hist(x, alpha=0.3)
                plt.xlim(-80, 80)
                plt.legend(label)
                plt.grid(True)

                plt.subplot(8, 2, i)
                plt.title('Image Feature [%s - %s]' % (roi, layer))
                plt.hist(y, alpha=0.3)
                plt.legend(label)
                plt.grid(True)

            i += 2

        plt.tight_layout()
        plt.savefig('plots\\results\\train_hist_%s.png' % layer)
        plt.close('all')

    elif layer == 'others':
        plt.figure(figsize=(20, 4))

        for other in others:
            loc = data[roi_up][other]

            for N in label:
                term = N[0].lower()

                x = loc[term + '_train']
                y = loc[term + '_y']

                plt.subplot(8, 2, i)
                plt.title('fMRI [%s - %s]' % (roi, layer))
                plt.hist(x, alpha=0.3)
                plt.xlim(-80, 80)
                plt.legend(label)
                plt.grid(True)

                plt.subplot(8, 2, i)
                plt.title('Image Feature [%s - %s]' % (roi, layer))
                plt.hist(y, alpha=0.3)
                plt.legend(label)
                plt.grid(True)

            i += 2

        plt.tight_layout()
        plt.savefig('plots\\results\\train_hist_%s.png' % layer)
        plt.close('all')


# =====================================================================================
# -------------------------------------------------------------------------------------
tool = Tools.Tool()

result = tool.read_result_data('s1')
cr_rate = tool.read_cr_rate('s1')
merged_data = tool.read_merged_data('s1')
xy_train_std = tool.read_xy_std_data('s1')
