
from layers.decoder_blocks import Layer as DecoderBlock
from tensorflow.keras.layers import Layer as BaseLayer


class Layer(BaseLayer):

    def __init__(self, kernel_size, in_channels, out_channels, **kwargs):
        super(Layer, self).__init__(**kwargs)
        self.db1 = DecoderBlock(kernel_size, in_channels, out_channels)
        self.db2 = DecoderBlock(kernel_size, in_channels, out_channels)
        self.db3 = DecoderBlock(kernel_size, in_channels, out_channels)
        self.db4 = DecoderBlock(kernel_size, in_channels, out_channels)
        self.db5 = DecoderBlock(kernel_size, in_channels, out_channels)
        self.db6 = DecoderBlock(kernel_size, in_channels, out_channels)
        self.db7 = DecoderBlock(kernel_size, in_channels, out_channels)

    def call(self, inputs, **kwargs):
        outputs = self.db1(inputs)
        outputs = self.db2(outputs)
        outputs = self.db3(outputs)
        outputs = self.db4(outputs)
        outputs = self.db5(outputs)
        outputs = self.db6(outputs)
        outputs = self.db7(outputs)
        return outputs


if __name__ == '__main__':
    import tensorflow as tf
    inputs = tf.random.normal((2, 4, 4, 512))
    layers = Layer(3, 64, 32)
    outputs = layers(inputs)
    print(outputs.shape)