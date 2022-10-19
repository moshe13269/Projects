
from abc import ABC
import tensorflow as tf


class SmoothL1Loss(tf.keras.losses.Loss, ABC):
    beta: float

    def __init__(self, beta: float, ):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta

    def call(self, y_true, y_pred):

        teacher_encoding, student_encoding = tf.split(y_pred, num_or_size_splits=2, axis=1)
        y_true = tf.expand_dims(y_true, axis=-1)
        tf.print(tf.shape(teacher_encoding), tf.shape(student_encoding), tf.shape(y_true))
        return tf.reduce_mean(teacher_encoding) - tf.reduce_mean(y_true)
        # z = tf.math.abs(teacher_encoding * y_true - student_encoding * y_true)
        # return tf.cond(z <= self.beta,
        #                lambda: tf.square(z) / (2 * self.beta),
        #                lambda: z - (self.beta / 2))
