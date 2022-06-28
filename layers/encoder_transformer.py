import tensorflow as tf
from tensorflow.keras.activations import gelu
from tensorflow.keras.layers import Layer as BaseLayer, Conv1D, LayerNormalization
from layers.encoder_attention import EncoderLayer


class Layer(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, rate=0.1):
        super(Layer, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_encoding = PositionalEncoding(channels=d_model)

        self.enc_layers = [
            EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, rate=rate)
            for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training=True, mask=None):
        x = self.pos_encoding(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, channels=512, kernel_size=3, stride=1):
        super(PositionalEncoding, self).__init__()
        self.conv1 = Conv1D(channels, kernel_size=kernel_size, strides=stride, padding='same')
        self.gn1 = LayerNormalization(axis=-1)

    def call(self, inputs):
        return gelu(self.gn1(self.conv1(inputs) + inputs))


if __name__ == '__main__':
    import tensorflow as tf
    inputs = tf.random.normal((2, 40, 512))
    layers = Layer(num_layers=24, d_model=512, num_heads=8, dff=4096)
    outputs = layers(inputs)
    print(outputs.shape)