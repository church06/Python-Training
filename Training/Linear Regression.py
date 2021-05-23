import math
import random

import matplotlib.pyplot as plt
import numpy
import sklearn.datasets
from numpy import ndarray


def boston_show(x: list, y):
    x = numpy.array(x)

    for i in range(0, 12):
        plt.subplot(3, 4, i + 1)
        plt.scatter(x[:, i], y)


def hwx(w1: float, x: float, w0: float):
    return w1 * x + w0


def accuracy(boolean_results: ndarray):
    boolean_results = numpy.array(boolean_results)

    counter_true = 0

    for i in boolean_results:
        if i:
            counter_true += 1

    return counter_true / boolean_results.size


def loss(x: list, y: list, fetch: int, alpha: float, test_x: list, test_y: list):
    w1 = random.uniform(0.0, 10)
    w0 = random.uniform(0.0, 10)

    x = numpy.array(x)
    y = numpy.array(y)

    numpy.array(test_x)
    numpy.array(test_y)

    results = []
    numpy.array(results)

    m = x.size

    if x.size != y.size:
        return -1
    else:
        for f in range(0, fetch):
            total = 0
            counter = 0
            pre_loss = 0

            for i in range(0, m):

                if pre_loss == total:
                    counter += 1

                y_lin = hwx(w1, x[i], w0)
                total += math.pow(y[i] - y_lin, 2)

                w1 = w1 + alpha * (y[i] - y_lin) * x[i]
                w0 = w0 + alpha * (y[i] - y_lin)

            print("fetch: ", f, " learning rate: ", alpha, " w1: ", w1, " w0: ", w0, " loss: ", total)

            plt.clf()
            plt.scatter(x, y)

            line = w0 + w1 * x
            plt.plot(x, line, color="r")
            plt.pause(0.01)


# =========================================================================================
file = sklearn.datasets.load_iris()

data = file.data
train_x = data[:120, 0]
train_y = data[:120, 2]

test_x = data[120:, 0]
test_y = data[120:, 2]

loss(train_x, train_y, 1000, 0.001, test_x, test_y)
