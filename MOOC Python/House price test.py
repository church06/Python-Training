import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

boston_housing = tf.keras.datasets.boston_housing
(train_x, train_y), (test_x, test_y) = boston_housing.load_data(test_split=0.2)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

titles = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B-1000', 'LSTAT', 'MEDV']
plt.figure(figsize=(12, 12))

for i in range(13):
    plt.subplot(4, 4, (i+1))
    plt.scatter(train_x[:, i],  train_y)

    plt.xlabel(titles[i])
    plt.ylabel("Price($1000's)")
    plt.title(str(i+1)+'.'+titles[i]+' - Price')
plt.show()