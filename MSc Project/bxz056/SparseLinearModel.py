import random

import numpy


def linear_regression(x: numpy.ndarray, voxels: int):
    wi = numpy.array([])
    w0 = random.random()

    for i in range(voxels):
        wi = numpy.append(wi, random.random())

    print('wi: ', wi.shape)
    y = x * wi + w0

    return numpy.array(y).reshape(voxels)


def gaussian_noise(linear_result):
    return linear_result + 1
