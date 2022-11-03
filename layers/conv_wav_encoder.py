import tensorflow as tf
from typing import List, Tuple
import tensorflow_addons as tfa
from tensorflow.python.keras.layers import Conv2D, Dropout, Dense, AveragePooling2D, Reshape


def clac_conv_output():
    pass


class ConvFeatureExtractionModel(tf.keras.layers.Layer):
    def __init__(self,
                 conv_layers: List[List[Tuple[int, int, int]]],
                 num_duplicate_layer: Tuple[int, int, int, int, int, int, int],
                 activation: str,
                 units: int,
                 dropout: float = 0.0,
                 mode: str = "default",
                 conv_bias: bool = False):
        super(ConvFeatureExtractionModel, self).__init__()
        self.conv_layers = None
        self.activation = tf.keras.layers.Activation(activation)
        self.reshape = Reshape((16, 512,))

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
                return tf.keras.Sequential([
                    make_conv0(),
                    Dropout(rate=dropout),
                    tf.keras.layers.LayerNormalization(),
                    tf.keras.layers.Activation(activation),
                    make_conv1(),
                    Dropout(rate=dropout),
                    tf.keras.layers.LayerNormalization(),
                ])

            elif is_group_norm and activation is not None:
                return tf.keras.Sequential([
                    make_conv0(),
                    Dropout(rate=dropout),
                    tfa.layers.GroupNormalization(),
                    tf.keras.layers.Activation(activation),
                    make_conv1(),
                    Dropout(rate=dropout),
                    tfa.layers.GroupNormalization(),
                ])

            else:
                return tf.keras.Sequential([make_conv0(),
                                            Dropout(rate=dropout),
                                            tf.keras.layers.Activation(activation),
                                            make_conv1(),
                                            Dropout(rate=dropout)])

        layers = []

        for i, layers_param in enumerate(conv_layers):
            assert len(layers_param) == 2 and len(layers_param[0]) == len(layers_param[1]) == 3, \
                "invalid conv definition: " + str(layers_param)
            # (dim, kernel, stride) = cl
            for j in range(num_duplicate_layer[i]):
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

        self.avg_pool = AveragePooling2D()

        self.fc = Dense(units=units, activation=activation)

    def call(self, x, **kwargs): #call(self, x):  #
        # BxT -> BxTxC

        for conv in self.conv_layers:
            inputs = x
            x = conv(inputs)
            x = self.activation(x)

        x = self.avg_pool(x)
        x = self.reshape(x)
        return self.fc(x)


if __name__ == '__main__':
    data = tf.random.normal((4, 32, 32, 3))
    conv_layers: List[List[Tuple[int, int, int]]] = [[(64, 3, 1), (64, 3, 1)],
                                                     [(128, 3, 2), (128, 3, 1)],
                                                     [(128, 3, 1), (128, 3, 1)],
                                                     [(256, 3, 1), (256, 3, 1)],
                                                     [(256, 3, 1), (256, 3, 1)],
                                                     [(512, 3, 2), (512, 3, 1)],
                                                     [(512, 3, 1), (512, 3, 1)]]

    num_duplicate_layer: Tuple[int, int, int, int, int, int, int] = (2, 1, 2, 1, 2, 1, 2)
    conv = ConvFeatureExtractionModel(conv_layers=conv_layers, activation='gelu', units=512,
                                      num_duplicate_layer=num_duplicate_layer)
    output = conv(data)
    print(output.shape)
    print(tf.reduce_mean(output), tf.reduce_min(output), tf.reduce_max(output))
