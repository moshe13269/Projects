
from layers.encoder_blocks import Layer as EncoderBlock
from tensorflow.keras.layers import Layer as BaseLayer


class Layer(BaseLayer):

    def __init__(self, kernel_size, stride_size, in_channels, out_channels, **kwargs):
        super(Layer, self).__init__(**kwargs)
        self.eb1 = EncoderBlock(kernel_size=kernel_size[0], strides=stride_size[0], out_channels=out_channels)
        self.eb2 = EncoderBlock(kernel_size=kernel_size[1], strides=stride_size[1], out_channels=out_channels)
        self.eb3 = EncoderBlock(kernel_size=kernel_size[2], strides=stride_size[2], out_channels=out_channels)
        self.eb4 = EncoderBlock(kernel_size=kernel_size[3], strides=stride_size[3], out_channels=out_channels)
        self.eb5 = EncoderBlock(kernel_size=kernel_size[4], strides=stride_size[4], out_channels=out_channels)
        self.eb6 = EncoderBlock(kernel_size=kernel_size[5], strides=stride_size[5], out_channels=out_channels)
        self.eb7 = EncoderBlock(kernel_size=kernel_size[6], strides=stride_size[6], out_channels=out_channels)

    def call(self, input, **kwargs):
        output = self.eb1(input)
        output = self.eb2(output)
        output = self.eb3(output)
        output = self.eb4(output)
        output = self.eb5(output)
        output = self.eb6(output)
        output = self.eb7(output)
        return output


if __name__ == '__main__':
    import tensorflow as tf
    inputs = tf.random.normal((2, 8000, 1))
    layers = Layer(kernel_size=(10, 5, 3, 3, 2, 2, 2), stride_size=(5, 3, 2, 2, 1, 1, 1), in_channels=1,
                   out_channels=512)
    outputs = layers(inputs)
    print(outputs.shape)
