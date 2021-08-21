import h5py
import numpy
from matplotlib import pyplot as plt


def time_plot(none, z, min_max, decimal):
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


# -------------------------------------------------------------------------------------
data_dir = 'G:\\Entrance\\Coding_Training\\PythonProgram\\MSc Project\\bxz056\\results.hdf5'

dataset = read_data(data_dir)

none_iter_200 = dataset['cnn1']['iter_200']['none']
z_iter_200 = dataset['cnn1']['iter_200']['z-score']
m_iter_200 = dataset['cnn1']['iter_200']['min-max']
d_iter_200 = dataset['cnn1']['iter_200']['decimal']

# none_iter_150 = dataset['cnn1']['iter_150']['none']
# z_iter_150 = dataset['cnn1']['iter_150']['z-score']
# m_iter_150 = dataset['cnn1']['iter_150']['min-max']
# d_iter_150 = dataset['cnn1']['iter_150']['decimal']
#
# none_iter_100 = dataset['cnn1']['iter_100']['none']
# z_iter_100 = dataset['cnn1']['iter_100']['z-score']
# m_iter_100 = dataset['cnn1']['iter_100']['min-max']
# d_iter_100 = dataset['cnn1']['iter_100']['decimal']
#
# none_iter_50 = dataset['cnn1']['iter_50']['none']
# z_iter_50 = dataset['cnn1']['iter_50']['z-score']
# m_iter_50 = dataset['cnn1']['iter_50']['min-max']
# d_iter_50 = dataset['cnn1']['iter_50']['decimal']
