
import tensorflow as tf
from tensorflow import keras
from typing import List, Tuple
import tensorflow_addons as tfa
from tensorflow.python.keras.layers import Conv2D, Dropout


class ConvFeatureExtractionModel(tf.keras.Model):
    def __init__(self,
                 conv_layers: List[List[Tuple[int, int, int]]], #List[Tuple[int, int, int]],
                 activation: str,
                 dropout: float = 0.0,
                 mode: str = "default",
                 conv_bias: bool = False):
        super(ConvFeatureExtractionModel, self).__init__()
        self.conv_layers = None
        self.activation = tf.keras.layers.Activation(activation)

        def block(layers_param,
                  activation,
                  is_layer_norm=False,
                  is_group_norm=False,
                  conv_bias=False):

            (dim0, kernel0, stride0) = layers_param[0]
            (dim1, kernel1, stride1) = layers_param[1]

            def make_conv0():
                conv = Conv2D(filters=dim0,
                              kernel_size=kernel0,
                              strides=stride0,
                              use_bias=conv_bias,
                              padding='same',
                              kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.))
                return conv

            def make_conv1():
                conv = Conv2D(filters=dim1,
                              kernel_size=kernel1,
                              strides=stride1,
                              use_bias=conv_bias,
                              padding='same',
                              kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.))
                return conv

            assert (is_layer_norm and is_group_norm) == False, "layer norm and group norm are exclusive"

            if is_layer_norm and activation is not None:
                return keras.Sequential([
                    make_conv0(),
                    Dropout(rate=dropout),
                    tf.keras.layers.LayerNormalization(),
                    tf.keras.layers.Activation(activation),
                    make_conv1(),
                    Dropout(rate=dropout),
                    tf.keras.layers.LayerNormalization(),
                ])

            elif is_group_norm and activation is not None:
                return keras.Sequential([
                    make_conv0(),
                    Dropout(rate=dropout),
                    tfa.layers.GroupNormalization(),
                    tf.keras.layers.Activation(activation),
                    make_conv1(),
                    Dropout(rate=dropout),
                    tfa.layers.GroupNormalization(),
                ])

            else:
                return keras.Sequential([make_conv0(),
                                         Dropout(rate=dropout),
                                         tf.keras.layers.Activation(activation),
                                         make_conv1(),
                                         Dropout(rate=dropout)])

        layers = []

        for i, layers_param in enumerate(conv_layers):
            assert len(layers_param) == 2 and len(layers_param[0]) == len(layers_param[1]) == 3, \
                "invalid conv definition: " + str(layers_param)
            # (dim, kernel, stride) = cl

            layers.append(
                block(
                    layers_param,
                    activation,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )

        self.conv_layers = layers

    def __call__(self, inputs):
        # BxT -> BxTxC
        inputs = tf.expand_dims(inputs, axis=-1)

        for conv in self.conv_layers:
            x = inputs
            inputs = conv(inputs)
            print(x.shape, inputs.shape)
            if inputs.shape == x.shape:
                inputs = self.activation(x + inputs)
            else:
                inputs = self.activation(inputs)
        return inputs


if __name__ == '__main__':
    data = tf.random.normal((2, 128, 128))
    conv_layers: List[List[Tuple[int, int, int]]] = [[(512, 3, 1), (512, 3, 1)],
                                                                           [(512, 3, 1), (512, 3, 1)],
                                                                           [(512, 3, 1), (512, 3, 1)],
                                                                           [(512, 3, 1), (512, 3, 1)],
                                                                           [(512, 3, 1), (512, 3, 1)],
                                                                           [(512, 3, 1), (512, 3, 1)],
                                                                           [(512, 3, 1), (512, 3, 1)]]
    conv = ConvFeatureExtractionModel(conv_layers=conv_layers, activation='gelu')
    output = conv(data)
    print(output.shape)
    print(tf.reduce_mean(output), tf.reduce_min(output), tf.reduce_max(output))
