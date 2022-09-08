
from tensorflow.python.keras.layers import Layer as BaseLayer, Conv1D, BatchNormalization
from tensorflow.python.keras.activations import gelu


class Layer(BaseLayer):

    def __init__(self, kernel_size, strides, out_channels, padding='same', **kwargs):
        super(Layer, self).__init__(**kwargs)
        self.conv1 = Conv1D(out_channels, kernel_size=kernel_size, strides=strides, padding=padding)
        self.gn1 = BatchNormalization(axis=-1)
        # self.conv2 = Conv1D(out_channels, kernel_size=kernel_size, padding=padding, strides=(2, 2))
        # self.gn2 = BatchNormalization(axis=-1)

    def call(self, inputs, **kwargs):
        outputs = gelu(self.gn1(self.conv1(inputs)))
        # outputs = gelu(self.gn2(self.conv2(outputs)))
        return outputs


if __name__ == '__main__':
    import tensorflow as tf
    inputs = tf.random.normal((2, 64, 64, 64))
    layers = Layer(3, 64, 32)
    outputs = layers(inputs)
    print(outputs.shape)