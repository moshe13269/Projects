import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Reshape, Flatten, ReLU
from typing import List, Tuple


class LinearClassifier(tf.keras.layers.Layer):
    """
    Transformer Encdoer outputs are: (batch, t', channels)
    Given N classes (types of parameters) when all specific class contain num_classes
    The Dense layer mapping latent space to accurate outputs:
    outputs -> class_i: (,t', channels) -> (, num_classes) and softmax

    """

    outputs_dimension_per_outputs: List[int]
    activation: str = 'softmax'
    dropout: float = 0.1

    def __init__(self,
                 outputs_dimension_per_outputs,
                 activation: str = 'softmax',
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)

        def make_layers():
            layers = []
            for output_dim in self.outputs_dimension_per_outputs:
                layers.append(
                    tf.keras.Sequential([
                        ReLU(),
                        Dropout(rate=dropout),
                        Flatten(),
                        Dense(units=output_dim, activation='relu'),
                        Dropout(rate=dropout),
                        # tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1)),
                        Dense(units=output_dim, activation='relu'),
                        Dense(units=output_dim, activation=None),
                    ])
                )
                # layers.append(Dense(units=output_dim, activation=self.activation))
            return layers

        self.outputs_dimension_per_outputs = outputs_dimension_per_outputs
        self.activation = activation
        self.layers = make_layers()

    def call(self, inputs, **kwargs):

        outputs = []

        for layer in self.layers:
            output = layer(inputs)
            # output = tf.nn.softmax(output - tf.math.reduce_max(output, keepdims=True, axis=-1))
            # output = tf.math.log(tf.nn.softmax(output) + 10**-1)
            outputs.append(output)

        return outputs
