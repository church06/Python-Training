from itertools import product

import h5py
import matplotlib.pyplot as plt
import numpy as np


def main():
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
