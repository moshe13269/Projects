from abc import ABC
import tensorflow as tf


class L2LossReg(tf.keras.losses.Loss, ABC):

    def __init__(self):
        super(L2LossReg, self).__init__()
        self.loss = tf.keras.losses.MeanSquaredError()

    def call(self, y_true, y_pred):
        y_true = tf.squeeze(y_true, axis=1)
        # tf.print(tf.shape(y_pred))
        # return tf.reduce_mean(y_true)-tf.reduce_mean(y_pred)
        return (y_true - y_pred)**2
