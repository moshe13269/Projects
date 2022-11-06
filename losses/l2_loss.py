
from abc import ABC
import tensorflow as tf


class L2Loss(tf.keras.losses.Loss, ABC):

    def __init__(self):
        super(L2Loss, self).__init__()
        self.loss = tf.keras.losses.MeanSquaredError()

    def call(self, y_true, y_pred):
        # return tf.reduce_mean(y_true) - tf.reduce_mean(y_pred)
        teacher_encoding, student_encoding = tf.split(y_pred, num_or_size_splits=2, axis=0)
        y_true = tf.expand_dims(y_true, axis=-1)
        return self.loss(teacher_encoding * y_true, student_encoding * y_true)
        # z = tf.math.abs(teacher_encoding * y_true - student_encoding * y_true)
        # return tf.reduce_mean(z)
        # return tf.cond(z <= self.beta,
        #                lambda: tf.square(z) / (2 * self.beta),
        #                lambda: z - (self.beta / 2))
