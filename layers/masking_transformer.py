import tensorflow as tf


class MaskingTransformer(tf.keras.layers.Layer):

    def __init__(self,
                 percent2mask):
        super(MaskingTransformer, self).__init__()
        self.percent2mask = percent2mask

    @tf.function
    def random_mask(self, inputs):
        mask = 1. - tf.random.uniform((tf.shape(inputs)[0], tf.shape(inputs)[1], 1)) // self.percent2mask
        return mask

    def call(self, inputs, **kwargs):
        mask = self.random_mask(inputs)

        return mask * inputs
