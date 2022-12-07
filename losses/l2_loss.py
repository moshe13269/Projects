from abc import ABC
import tensorflow as tf


class L2Loss(tf.keras.losses.Loss, ABC):

    def __init__(self):
        super(L2Loss, self).__init__()
        self.loss = tf.keras.losses.MeanSquaredError()
        self.subtract = tf.keras.layers.Subtract()
        self.mul = tf.keras.layers.Multiply()

    def call(self, y_true, y_pred):
        student, teacher = tf.split(y_pred, num_or_size_splits=2, axis=0)
        # return self.subtract(y_true - y_pred)**2
        x = self.subtract([student, teacher])
        return tf.reduce_mean(self.mul([x, x]))
