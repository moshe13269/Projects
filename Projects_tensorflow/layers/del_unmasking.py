
import tensorflow as tf


class DelUnMasking(tf.keras.layers.Layer):

    def __init__(self):
        super(DelUnMasking, self).__init__()

        self.add = tf.keras.layers.Add()
        self.subtract = tf.keras.layers.Subtract()
        self.mul = tf.keras.layers.Multiply()

    def call(self, data, **kwargs):
        latent_z, mask = data
        mask = tf.expand_dims(mask, axis=-1)

        latent_z_masked = self.add([self.mul([mask, self.learnable_mask]),
                                    self.mul([tf.subtract(1., mask), latent_z])])

        return latent_z_masked

    @tf.function
    def better_loss(self, mask, inputs):
        mask = tf.cast(tf.logical_or(y_true >= 0.1, y_pred >= 0.1), tf.float32)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)