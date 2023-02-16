import tensorflow as tf
from tensorflow import keras
from typing import List, Tuple
import tensorflow_addons as tfa
from tensorflow.python.keras.layers import Conv1D, Dropout


class ConvFeatureExtractionModel(tf.keras.Model):
    def __init__(self,
                 # conv_inputs_shape: List[Tuple[int, int, int]],
                 # conv_outputs_shape: List[Tuple[int, int, int]],
                 conv_layers: List[Tuple[int, int, int]],
                 dropout: float = 0.0,
                 mode: str = "default",
                 conv_bias: bool = False):
        super(ConvFeatureExtractionModel, self).__init__()
        self.conv_layers = None

        def block(n_out,
                  kernel,
                  strides,
                  index_layer,
                  is_layer_norm=False,
                  is_group_norm=False,
                  conv_bias=False):

            # @tf.function
            def make_conv():
                conv = Conv1D(filters=n_out, kernel_size=kernel, strides=strides, use_bias=conv_bias,
                              kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.))
                return conv

            # def build_layer(layer):
            #     if index_layer == 0:
            #         layer.build(inputs_shape)
            #     else:
            #         layer.build(outputs_shape)
            #     return layer

            assert (is_layer_norm and is_group_norm) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                layer = tf.keras.Sequential([
                    make_conv(),
                    Dropout(rate=dropout),
                    tf.keras.layers.LayerNormalization(),
                    tf.keras.layers.Activation(tf.nn.gelu),
                ])
                return layer #build_layer(layer)

            elif is_group_norm:
                layer = tf.keras.Sequential([
                    make_conv(),
                    Dropout(rate=dropout),
                    tfa.layers.GroupNormalization(),
                    tf.keras.layers.Activation(tf.nn.gelu),  # tf.keras.activations.gelu(),
                ])
                return layer #build_layer(layer)

            else:
                layer = tf.keras.Sequential([make_conv(), Dropout(rate=dropout), tf.keras.layers.Activation(tf.nn.gelu)])
                return layer #build_layer(layer)

        layers = []

        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, kernel, stride) = cl

            layers.append(
                block(
                    dim,
                    kernel,
                    stride,
                    i,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )

        self.conv_layers = layers

    def call(self, x): #def call(self, inputs):
        # BxT -> BxTxC
        if tf.shape(x) == 3:
            inputs = tf.expand_dims(x, axis=-1)

        for conv in self.conv_layers:
            inputs = conv(inputs)

        return inputs


if __name__ == '__main__':
    data = tf.random.normal((2, 1024))
    conv_layers: List[Tuple[int, int, int]] = [(512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 2, 1),
                                               (512, 2, 1), (512, 2, 1)]
    conv_inputs_shape: Tuple[int, int, int] = (32, 32, 512,)
    conv_outputs_shape: Tuple[int, int, int] = ()
    conv = ConvFeatureExtractionModel(conv_layers=conv_layers)
    output = conv(data)
    print(output.shape)
