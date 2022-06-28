
import tensorflow as tf


class Layer(tf.keras.layers.Layer):
    def __init__(self, *, word_depth=512):
        super(Layer, self).__init__()

        self.word_depth = word_depth
        self.learnable_mask = self.add_weight("learnable_mask", shape=(1, 1, self.word_depth), trainable=True)

    def call(self, data, **kwargs):
        latent_z, mask = data

        mask = tf.expand_dims(mask, axis=-1)
        # mask = tf.ones_like(inputs) * mask
        masked = mask * self.learnable_mask + tf.multiply(1. - mask, latent_z)

        return masked


if __name__ == '__main__':
    import tensorflow as tf
    inputs = [tf.random.normal((2, 40, 6)), tf.math.round(tf.random.uniform(shape=(2, 40), maxval=1))]
    layers = Layer(word_depth=6)
    outputs = layers(inputs)
    print(outputs) #.shape)
