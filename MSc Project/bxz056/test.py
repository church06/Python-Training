from itertools import product

import h5py
import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import tensorflow
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import Tools


def main():
    nan_test()


def nan_test():
    tool = Tools.Tool()
    data = tool.read_merged_data('s1')

    for R in ['v1', 'v2', 'v3', 'v4', 'loc', 'ffa', 'ppa']:
        X = data[R.upper()]['hmax2']['min-max']['cor_pt_av']

        print(R + ' ----------------------------')
        print(np.linalg.matrix_rank(X))
        print(X)


def str_test():
    test = 'Af;oinsdf'

    print(test[0].lower())


def tf_computation():
    test = numpy.random.randint(1, 10, size=(1, 10))
    num_1 = 2
    num_2 = 2
    tf_test_1 = tensorflow.math.divide(test, num_1)
    tf_test_2 = tensorflow.math.multiply(test, num_2)

    print(test)
    print(tf_test_1)
    print(tf_test_2)
    print(numpy.array(tf_test_1))


def threshold_test():
    test = numpy.random.rand(1, 10)

    type_list = test > 5
    print('test: ', test)
    print('type: ', type_list)
    print('sum: ', numpy.sum(type_list))


def numpy_function_test():
    print(
        numpy.subtract(
            numpy.multiply(
                numpy.divide(
                    numpy.subtract(3, 1), numpy.subtract(11, 1)),
                2),
            1)
    )


def scaler_test():
    file = h5py.File('HDF5s\\xy_train&std_data.hdf5', 'r')
    dataset = numpy.array(file['s1']['V1']['cnn1']['n_train'])
    file.close()

    list_1 = numpy.array(dataset[:5, :3], dtype='float64')
    list_2 = numpy.array(dataset[:5, 3:6], dtype='float64')

    scaler_1 = MinMaxScaler(feature_range=(-1, 1))
    s_list_1 = scaler_1.fit_transform(list_1)

    scaler_2 = StandardScaler(with_mean=False)
    s_list_2 = scaler_2.fit_transform(list_2)

    print('list_1: ', list_1, '\n')
    print('scaled: ', s_list_1, '\n')
    print('transfer: ', scaler_1.inverse_transform(s_list_1), '\n')
    print('transfer_origin: ', scaler_1.inverse_transform(list_1), '\n')

    print('list_2: ', list_2, '\n')
    print('scaled: ', s_list_2, '\n')
    print('transfer: ', scaler_2.inverse_transform(s_list_2), '\n')


def read_test():
    file = h5py.File('G:\\Entrance\\Coding_Training\\PythonProgram\\MSc Project\\bxz056\\HDF5s\\test.hdf5', 'a')

    test = {}
    test_2 = {}

    for i in range(0, 100):
        test_2[str(i)] = i

    test['test'] = test_2

    key_1 = list(test.keys())
    key_2 = list(test[key_1[0]].keys())

    print('key_1: ', key_1)
    print('key_2', key_2)

    for k_1, k_2 in product(key_1, key_2):
        try:
            loc = file.create_group(k_1)

        except ValueError:
            loc = file[k_1]

        loc.create_dataset(k_2, data=test[k_1][k_2])

    file.close()


def shape_test_data_na():
    test = np.ones((2, 3))
    print('test: ', test.shape)

    test = test.T
    print('test after T: ', test.shape)


def select_top_test_data_na():
    axis = 1
    data = np.array([(1, 2, 3, 4, 5, 6, 7), (1, 2, 3, 4, 5, 6, 7)])
    value = np.array([(1, 2, 3, 4, 5, 6, 7), (1, 2, 3, 4, 5, 6, 7)])
    num = 500

    num_elem = data.shape[axis]

    sorted_index = np.argsort(value)[::-1]

    rank = np.zeros(num_elem, dtype=int)
    rank[sorted_index] = np.array(range(0, num_elem))

    selected_index_bool = rank < num

    print('num: ', num)
    print('rank: ', rank)
    print('selected_index_bool: ', selected_index_bool)

    if axis == 0:
        selected_index = np.array(range(0, num_elem), dtype=int)[selected_index_bool]

    elif axis == 1:
        selected_index = np.array(range(0, num_elem), dtype=int)[selected_index_bool]
    else:
        selected_index = np.array([])

    print(selected_index.shape)


def zero_mean_gaussian_random_test():
    mean = [0]
    num = 1000
    conv = [[1]]
    matrix = np.random.multivariate_normal(mean, conv, num)
    index = np.array(range(0, num))

    matrix = matrix.reshape(num)
    print(matrix.shape)
    print(index.shape)

    plt.title('None absolute')
    plt.subplot(211)
    plt.bar(index, matrix)

    plt.title('Absolute')
    plt.subplot(212)
    plt.bar(index, np.abs(matrix))

    plt.tight_layout()
    plt.show()


# ===================================
main()
