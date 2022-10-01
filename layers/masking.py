
import tensorflow as tf


class Masking(tf.keras.layers.Layer):

    num_channels: int

    def __init__(self, *, num_channels: int):
        super(Masking, self).__init__()

        self.num_channels = num_channels
        self.learnable_mask = self.add_weight("learnable_mask", shape=(1, 1, self.num_channels), trainable=True)

    def call(self, data, **kwargs):
        latent_z, mask = data

        # mask = tf.expand_dims(mask, axis=-1)
        # mask = tf.ones_like(inputs) * mask
        latent_z_masked = mask * self.learnable_mask + tf.multiply(1. - mask, latent_z)

        return latent_z_masked, mask


if __name__ == '__main__':
    import tensorflow as tf
    inputs = [tf.random.normal((2, 40, 6)), tf.math.round(tf.random.uniform(shape=(2, 40), maxval=1))]
    layers = Masking(num_channels=6)
    outputs = layers(inputs)
    print(outputs)
