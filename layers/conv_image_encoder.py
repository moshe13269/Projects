import tensorflow as tf
from tensorflow import keras
from typing import List, Tuple
import tensorflow_addons as tfa
from tensorflow.python.keras.layers import Conv2D, Dropout, MaxPool2D


class CreatePatches(tf.keras.layers.Layer):

    def __init__(self, patch_size):
        super(CreatePatches, self).__init__()
        self.patch_size = 32
        self.resnet = tf.keras.applications.resnet50.ResNet50(input_shape=(32, 32, 3), include_top=False)

    def call(self, inputs):
        patches = []
        #For square images only (as inputs.shape[1] = inputs.shape[2])
        input_image_size = inputs.shape[1]
        for i in range(0, input_image_size, self.patch_size):
            for j in range(0, input_image_size, self.patch_size):
                patches.append(self.resnet(inputs[:, i: i + self.patch_size, j: j + self.patch_size, :]))
        return patches










class ConvFeatureExtractionModel(tf.keras.Model):
    def __init__(self,
                 conv_layers: List[Tuple[int, int, int]],
                 dropout: float = 0.0,
                 mode: str = "default",
                 conv_bias: bool = False):
        super(ConvFeatureExtractionModel, self).__init__()
        self.conv_layers = None

        def block(n_out,
                  kernel,
                  strides,
                  is_layer_norm=False,
                  is_group_norm=False,
                  conv_bias=False):

            def make_conv():
                conv = Conv2D(filters=n_out,
                              kernel_size=kernel,
                              strides=strides,
                              use_bias=conv_bias,
                              padding="same",
                              kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.))
                return conv

            assert (is_layer_norm and is_group_norm) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return keras.Sequential([
                    make_conv(),
                    Dropout(rate=dropout),
                    tf.keras.layers.LayerNormalization(),
                    tf.keras.layers.Activation(tf.nn.relu),
                ])
            elif is_group_norm:
                return keras.Sequential([
                    make_conv(),
                    Dropout(rate=dropout),
                    tfa.layers.GroupNormalization(),
                    tf.keras.layers.Activation(tf.nn.relu),  # tf.keras.activations.gelu(),
                ])
            else:
                return keras.Sequential([make_conv(), Dropout(rate=dropout), tf.keras.layers.Activation(tf.nn.relu)])

        layers = []

        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, kernel, stride) = cl

            layers.append(
                block(
                    dim,
                    kernel,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )

        self.conv_layers = layers

    def call(self, inputs, **kwargs):
        # BxT -> BxTxC
        if len(inputs.shape) == 3:
            inputs = tf.expand_dims(inputs, axis=-1)

        for conv in self.conv_layers:
            inputs = conv(inputs)

        return inputs


if __name__ == '__main__':
    data = tf.random.normal((2, 1024))
    conv_layers: List[Tuple[int, int, int]] = [(64, 3, 1), (64, 3, 1),
                                               (128, 3, 1), (128, 3, 1),
                                               (256, 2, 1), (256, 2, 1), (256, 2, 1), (256, 2, 1),
                                               (512, 2, 1), (512, 2, 1), (512, 2, 1), (512, 2, 1), (512, 2, 1),
                                               (512, 2, 1), (512, 2, 1)]
    conv = ConvFeatureExtractionModel(conv_layers=conv_layers)
    output = conv(data)
    print(output.shape)
