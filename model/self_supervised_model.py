import tensorflow as tf
from layers.masking import Layer as Masking
from layers.quantizer import Layer as Quantizer
from layers.encoder_latent_space import Layer as Encoder
from layers.encoder_transformer_2 import Layer as Transformer


class Wav2Vec(tf.keras.Model):

    def __init__(self, params):
        super().__init__()

        self.encoder = Encoder(kernel_size=(10, 5, 3, 3, 2, 2, 2), stride_size=(5, 3, 2, 2, 1, 1, 1), in_channels=1,
                               out_channels=512)
        self.mask = Masking(word_depth=512)
        self.quantizer = Quantizer(G=2, V=320, activation=None, tau=2.)
        self.transformer = Transformer(num_layers=24, d_model=512, num_heads=8, dff=4096)

    def call(self, inputs, training=True):
        latent_space = self.encoder(inputs[0])

        p_g_v, q_t = self.quantizer(latent_space)

        c_t = self.transformer(self.mask([latent_space, inputs[1]]))

        c_t = tf.concat([q_t, c_t], axis=-1)

        return c_t, p_g_v


if __name__ == '__main__':
    model = Wav2Vec
    c_t, p_g_v = model(
        (tf.random.normal(shape=(10, 8000, 1)), tf.math.round(tf.random.uniform(shape=(10, 134), maxval=1))))
    print(c_t.shape, p_g_v.shape)
