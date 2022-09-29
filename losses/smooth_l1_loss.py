from abc import ABC

import tensorflow as tf


class SmoothL1Loss(tf.keras.losses.Loss, ABC):
    beta: float

    def __init__(self, beta: float, ):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta

    def call(self, y_true, y_pred):
        z = tf.math.abs(y_true - y_pred)
        return tf.cond(z <= self.beta,
                       lambda: tf.square(z) / (2 * self.beta),
                       lambda: z - (self.beta / 2))
