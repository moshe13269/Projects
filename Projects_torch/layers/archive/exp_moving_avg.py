
import tensorflow as tf


class EMA(tf.keras.layers.Layer):

    tau: int

    def __init__(self, tau: int):
        super(EMA, self).__init__()

        self.tau = tau

    def call(self, inputs, **kwargs):
        return self.tau
