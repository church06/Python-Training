import matplotlib.pyplot as plt
import numpy as np


def main():
    zero_mean_gaussian_random_test()


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
