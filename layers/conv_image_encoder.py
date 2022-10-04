
import tensorflow as tf
from tensorflow import keras
from typing import List, Tuple
import tensorflow_addons as tfa
from tensorflow.python.keras.layers import Conv1D, Dropout


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
                conv = Conv1D(filters=n_out, kernel_size=kernel, strides=strides, use_bias=conv_bias,
                              kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.))
                return conv

            assert (is_layer_norm and is_group_norm) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return keras.Sequential([
                    make_conv(),
                    Dropout(rate=dropout),
                    tf.keras.layers.LayerNormalization(),
                    tf.keras.layers.Activation(tf.nn.gelu),
                ])
            elif is_group_norm:
                return keras.Sequential([
                    make_conv(),
                    Dropout(rate=dropout),
                    tfa.layers.GroupNormalization(),
                    tf.keras.layers.Activation(tf.nn.gelu),  # tf.keras.activations.gelu(),
                ])
            else:
                return keras.Sequential([make_conv(), Dropout(rate=dropout), tf.keras.layers.Activation(tf.nn.gelu)])

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

    def __call__(self, inputs):
        # BxT -> BxTxC
        inputs = tf.expand_dims(inputs, axis=-1)

        for conv in self.conv_layers:
            inputs = conv(inputs)

        return inputs


if __name__ == '__main__':
    data = tf.random.normal((2, 1024))
    conv_layers: List[Tuple[int, int, int]] = [(512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 2, 1),
                                               (512, 2, 1), (512, 2, 1)]
    conv = ConvFeatureExtractionModel(conv_layers=conv_layers)
    output = conv(data)
    print(output.shape)
