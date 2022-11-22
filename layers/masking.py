
import tensorflow as tf


class Masking(tf.keras.layers.Layer):

    num_channels: int

    def __init__(self, num_channels: int):
        super(Masking, self).__init__()

        self.learnable_mask = None
        self.num_channels = num_channels
        self.add = tf.keras.layers.Add()
        self.subtract = tf.keras.layers.Subtract()
        self.mul = tf.keras.layers.Multiply()

    def build(self, input_shape):
        self.learnable_mask = self.add_weight(shape=(1, 1, self.num_channels),
                                              trainable=True,
                                              initializer='random_normal',
                                              name='masking')

    def call(self, data, **kwargs):
        latent_z, mask = data
        mask = tf.expand_dims(mask, axis=-1)
        latent_z_masked = self.add([self.mul([mask, self.learnable_mask]),
                                    self.mul([self.subtract([1., mask]), latent_z])])

        return latent_z_masked


if __name__ == '__main__':
    import tensorflow as tf
    inputs = [tf.Variable(tf.random.normal((2, 10000, 8))), tf.Variable(tf.random.uniform(shape=(2, 40), maxval=1))]
    layers = Masking(num_channels=8)
    layers.build((None, 40))
    outputs = layers(inputs)
    print(outputs[0].shape)
    a=0
