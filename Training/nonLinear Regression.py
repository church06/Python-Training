import math
import random

import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.datasets


def hypothesis(w0, x_, w1, w2, w3, w4):
    output = round(w0 + w1 * x_
                   + w2 * math.pow(x_, 2)
                   + w3 * math.pow(x_, 3)
                   + w4 * math.pow(x_, 4), 3)

    return output


def loss(y: float, hwx: float):
    cost = y - hwx
    return math.pow(cost, 2)


def gradient_descent(w, alpha, y, hwx, x, sqr):
    return w + alpha * (y - hwx) * math.pow(x, sqr)


def non_linear_regression(x_: numpy.ndarray, y_: numpy.ndarray,
                          learning_rate: float, fetch: int):
    m = x_.size

    w0 = random.uniform(0, 1)
    w1 = random.uniform(0, 1)
    w2 = random.uniform(0, 1)
    w3 = random.uniform(0, 1)
    w4 = random.uniform(0, 1)

    for f in range(0, fetch):

        lose = 0
        hwx_list = numpy.array([])

        for i in range(0, m):
            x = x_[i, 0]
            y = y_[i, 0]

            hwx = hypothesis(w0, x, w1, w2, w3, w4)

            lose += (1 / (2 * m)) * loss(y, hwx)

            hwx_list = numpy.append(hwx_list, hwx)

            w0 = gradient_descent(w0, learning_rate, y, hwx, 1, 0)
            w1 = gradient_descent(w1, learning_rate, y, hwx, x, 1)
            w2 = gradient_descent(w2, learning_rate, y, hwx, x, 2)
            w3 = gradient_descent(w3, learning_rate, y, hwx, x, 3)
            w4 = gradient_descent(w4, learning_rate, y, hwx, x, 4)

        print("fetch: ", f,
              "w0:", w0,
              " w1:", w1,
              " w2:", w2,
              " w3:", w3,
              " w4:", w4,
              "lose:", lose)

        plt.clf()
        plt.scatter(x_, y_)

        plt.plot(x_, hwx_list, color='r')
        plt.pause(0.001)


file = sklearn.datasets.load_boston()

data = file.data
target = file.target

train_data = pandas.DataFrame(data, columns=file['feature_names'])
train_target = pandas.DataFrame(target, columns=['MEDV'])

train_x = train_data['CRIM']

train_x = numpy.array(train_x).reshape(train_x.size, 1)
train_target = numpy.array(train_target)

switch = 2

if switch == 0:
    plt.scatter(train_target, train_x)
elif switch == 1:
    non_linear_regression(train_x, train_target, 0.00000000001, 1000)
else:
    train_x = numpy.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]).reshape(14, 1)
    train_target = numpy.array([[1, 4, 9, 16, 25, 36, 49, 64, 68, 67, 49, 36, 25, 16]]).reshape(14, 1)

    non_linear_regression(train_x, train_target, 0.0000000001, 400)
