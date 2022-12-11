import tensorflow as tf
from typing import List, Tuple
import tensorflow_addons as tfa
from tensorflow.python.keras.layers import Conv1DTranspose, Dropout, Dense, AveragePooling1D, Reshape


class ConvDecoderModel(tf.keras.layers.Layer):
    def __init__(self,
                 conv_layers: List[Tuple[int, int, int]],
                 num_duplicate_layer: Tuple[int, int, int, int, int, int, int],
                 activation: str,
                 units: int,
                 dropout: float = 0.0,
                 mode: str = "default",
                 conv_bias: bool = False):
        super(ConvDecoderModel, self).__init__()
        self.conv_layers = None
        self.activation = tf.keras.layers.Activation(activation)

        def block(layers_param,
                  activation,
                  is_layer_norm=False,
                  is_group_norm=True,
                  conv_bias=False):

            (dim, kernel, stride, output_padding) = layers_param

            def make_conv():
                conv = Conv1DTranspose(filters=dim,
                                       kernel_size=kernel,
                                       strides=stride,
                                       use_bias=conv_bias,
                                       output_padding=output_padding,  # padding='same', #
                                       kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.))
                return conv

            assert (is_layer_norm and is_group_norm) == False, "layer norm and group norm are exclusive"

            if is_layer_norm and activation is not None:
                return tf.keras.Sequential([
                    make_conv(),
                    Dropout(rate=dropout),
                    tf.keras.layers.LayerNormalization(),
                    tf.keras.layers.Activation(activation),
                ])

            elif is_group_norm and activation is not None:
                return tf.keras.Sequential([
                    make_conv(),
                    Dropout(rate=dropout),
                    tfa.layers.GroupNormalization(),
                    tf.keras.layers.Activation(activation),
                ])

            else:
                return tf.keras.Sequential([
                    make_conv(),
                    Dropout(rate=dropout),
                    tf.keras.layers.Activation(activation),
                ])

        layers = []

        for i, layers_param in enumerate(conv_layers):

            for j in range(num_duplicate_layer[i]):
                layers.append(
                    block(
                        layers_param,
                        activation,
                        is_layer_norm=mode == "layer_norm",
                        is_group_norm=mode == "default",
                        conv_bias=conv_bias,
                    )
                )

        self.conv_layers = layers

        self.avg_pool = AveragePooling1D()

        self.fc = Dense(units=units, activation=None)

    def call(self, x, **kwargs):
        # BxT -> BxTxC

        for conv in self.conv_layers:
            x = conv(x)

        return self.fc(x)


if __name__ == '__main__':
    data = tf.random.normal((4, 50, 512))
    conv_layers: List[Tuple[int, int, int, int]] = [(512, 2, 2, 1),
                                                    (512, 2, 2, 1),
                                                    (512, 3, 2, 1),
                                                    (512, 3, 2, 1),
                                                    (512, 3, 2, 0),
                                                    (512, 3, 2, 0),
                                                    (512, 10, 5, 4)]

    num_duplicate_layer: Tuple[int, int, int, int, int, int, int] = (1, 1, 1, 1, 1, 1, 1)
    conv = ConvDecoderModel(conv_layers=conv_layers, activation='gelu', units=1,
                                      num_duplicate_layer=num_duplicate_layer)
    output = conv(data)
    print(output.shape)
    print(tf.reduce_mean(output), tf.reduce_min(output), tf.reduce_max(output))
