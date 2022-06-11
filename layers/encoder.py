
from layers.encoder_blocks import Layer as EncoderBlock
from tensorflow.keras.layers import Layer as BaseLayer


class Layer(BaseLayer):

    def __init__(self, kernel_size, in_channels, out_channels, **kwargs):
        super(Layer, self).__init__(**kwargs)
        self.eb1 = EncoderBlock(kernel_size, in_channels, out_channels)
        self.eb2 = EncoderBlock(kernel_size, in_channels, out_channels)
        self.eb3 = EncoderBlock(kernel_size, in_channels, out_channels)
        self.eb4 = EncoderBlock(kernel_size, in_channels, out_channels)
        self.eb5 = EncoderBlock(kernel_size, in_channels, out_channels)
        self.eb6 = EncoderBlock(kernel_size, in_channels, out_channels)
        self.eb7 = EncoderBlock(kernel_size, in_channels, out_channels)

    def call(self, inputs, **kwargs):
        outputs = self.eb1(inputs)
        outputs = self.eb2(outputs)
        outputs = self.eb3(outputs)
        outputs = self.eb4(outputs)
        outputs = self.eb5(outputs)
        outputs = self.eb6(outputs)
        outputs = self.eb7(outputs)
        return outputs


if __name__ == '__main__':
    import tensorflow as tf
    inputs = tf.random.normal((2, 512, 512, 2))
    layers = Layer(3, 64, 32)
    outputs = layers(inputs)
    print(outputs.shape)
