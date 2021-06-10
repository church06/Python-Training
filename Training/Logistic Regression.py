import math
import random

import matplotlib.pyplot as plt
import numpy
import sklearn.datasets


# TODO: 改算法，显示决策边界。

def hwx(w0: float, x_1: float, x_2: float, w1: float, w2: float, w3: float, w4: float):
    wtx = w0 + w1 * x_1 + w2 * x_2 + w3 * math.pow(x_1, 2) + w4 * math.pow(x_2, 2)

    return wtx


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
        if hw_x != 0:
            output = - math.log(hw_x, 2)
        else:
            return 0

    else:
        if hw_x == 1:
            return 1
        else:
            output = - math.log(1 - hw_x, 2)

    return output


def nonlinear_regression(train_x: numpy.ndarray, answer: numpy.ndarray, alpha, fetch):
    m = train_x[:, 0].size

    w0 = 1 - numpy.max(train_x)
    w1 = random.uniform(0, 1)
    w2 = random.uniform(0, 1)
    w3 = random.uniform(0, 1)
    w4 = random.uniform(0, 1)

    for f in range(0, fetch):
        result = 0

        result_1 = numpy.array([])

        for index in range(0, m):
            hwx_res = hwx(w0, train_x[index, 1], train_x[index, 3], w1, w2, w3, w4)

            result_1 = numpy.append(result_1, hwx_res)
            hwx_res = sigmoid(hwx_res)

            lost = cost(answer[i], hwx_res)

            result += (1 / (index + 1)) * lost

            w0 = gradient_descent(w0, alpha, answer[index], hwx_res, train_x[index, 1], 0)
            w1 = gradient_descent(w1, alpha, answer[index], hwx_res, train_x[index, 1], 1)
            w2 = gradient_descent(w2, alpha, answer[index], hwx_res, train_x[index, 3], 1)
            w3 = gradient_descent(w3, alpha, answer[index], hwx_res, train_x[index, 1], 2)
            w4 = gradient_descent(w4, alpha, answer[index], hwx_res, train_x[index, 3], 2)

        print("fetch: ", f, " w1: ", w1, " w2: ", w2, "w3: ", w3, "w4: ", w4, "lose: ", result)

        plt.clf()
        plt.scatter(train_x[:, 1], train_x[:, 3])

        plt.plot(result_1, result_1, color='r')
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
    if i == 0:
        trans_ans = numpy.append(trans_ans, 0)
    else:
        trans_ans = numpy.append(trans_ans, 1)

switch = 1

if switch == 1:
    x1_0 = iris_0[:, 1]
    x2_0 = iris_0[:, 3]

    x1_1 = iris_1[:, 1]
    x2_1 = iris_1[:, 3]

    x1_2 = iris_2[:, 1]
    x2_2 = iris_2[:, 3]

    plt.scatter(x1_0, x2_0, color='r')
    plt.scatter(x1_1, x2_1, color='g')
    plt.scatter(x1_2, x2_2, color='b')

else:
    nonlinear_regression(train_x=data, answer=trans_ans, alpha=0.000000000001, fetch=200)
