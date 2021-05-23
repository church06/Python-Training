import math
import random

import matplotlib.pyplot as plt
import numpy
import sklearn.datasets


def hwx(w0: float, x_1: float, x_2: float, w1: float, w2: float, w3: float):
    wtx = w0 + w1 * x_1 + w2 * x_2 + w3 * x_1 * x_2

    return sigmoid(wtx)


def sigmoid(z: float):
    gz = 1 / (1 + math.pow(math.e, -z))

    if gz > 0.5:
        return 1
    else:
        return 0


def gradient_descent(w: float, alpha: float, y: float, hwx: float, x: float, sqr: int):
    return w + alpha * (y - hwx) * math.pow(x, sqr)


def cost(y, hw_x):
    if y == 1:
        output = - math.log(hw_x, 2)

    else:
        if hw_x == 1:
            return 1
        else:
            output = - math.log(1 - hw_x, 2)

    return output


def nonlinear_regression(train_x: numpy.ndarray, answer: numpy.ndarray, alpha, fetch):
    m = train_x[:, 0].size

    w0 = 1
    w1 = random.uniform(1, 5)
    w2 = random.uniform(1, 5)
    w3 = random.uniform(1, 5)

    for f in range(0, fetch):
        result = 0

        result_list = []
        result_list = numpy.array(result_list)

        for index in range(0, m):
            hwx_res = hwx(w0, train_x[index, 1], train_x[index, 3], w1, w2, w3)

            result += (1 / (index + 1)) * cost(answer[i], hwx_res)
            result_list = numpy.append(result_list, hwx_res)

            w1 = gradient_descent(w1, alpha, answer[index], hwx_res, train_x[index, 1], 1)
            w2 = gradient_descent(w1, alpha, answer[index], hwx_res, train_x[index, 3], 1)
            w3 = gradient_descent(w1, alpha, answer[index], hwx_res, train_x[index, 1] * train_x[index, 3], 1)

        print("fetch: ", f, " w1: ", w1, " w2: ", w2, "w3: ", w3)

        plt.clf()
        plt.scatter(train_x[:, 1], train_x[:, 3])

        print(result_list)

        plt.plot(result_list, color='r')
        plt.pause(0.01)


file = sklearn.datasets.load_iris()

data = numpy.array(file.data)
target = numpy.array(file.target)

iris_0 = [[]]
iris_1 = [[]]
iris_2 = [[]]

iris_0 = numpy.array(iris_0)
iris_1 = numpy.array(iris_1)
iris_2 = numpy.array(iris_2)

iris0_counter = 0
iris1_counter = 0
iris2_counter = 0

for i in range(0, target.size):
    if target[i] == 0:
        iris0_counter += 1
        iris_0 = numpy.append(iris_0, data[i, :])

    elif target[i] == 1:
        iris1_counter += 1
        iris_1 = numpy.append(iris_1, data[i, :])

    else:
        iris2_counter += 1
        iris_2 = numpy.append(iris_2, data[i, :])

iris_0 = iris_0.reshape(iris0_counter, 4)
iris_1 = iris_1.reshape(iris1_counter, 4)
iris_2 = iris_2.reshape(iris2_counter, 4)

trans_ans = []
trans_ans = numpy.array(trans_ans)

for i in target:
    if i == 1:
        trans_ans = numpy.append(trans_ans, i)
    else:
        trans_ans = numpy.append(trans_ans, 0)

nonlinear_regression(train_x=data, answer=trans_ans, alpha=0.001, fetch=1000)
