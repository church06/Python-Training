import abc

import tensorflow as tf


class MyModel(tf.keras.Model, abc.ABC):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv2d_1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.d2 = tf.keras.layers.Dense(10)

    def call(self, inputs, training=None, mask=None):
        inputs = self.conv2d_1(inputs)
        inputs = self.flatten(inputs)
        inputs = self.d1(inputs)
        return self.d2(inputs)
