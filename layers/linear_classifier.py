import tensorflow as tf
from tensorflow.keras.layers import Dense
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

    def __init__(self,
                 outputs_dimension_per_outputs,
                 activation,
                 **kwargs):
        super().__init__(**kwargs)

        def make_layers():
            layers = []
            for output_dim in self.outputs_dimension_per_outputs:
                layers.append(Dense(units=output_dim, activation=self.activation))
            return layers

        self.outputs_dimension_per_outputs = outputs_dimension_per_outputs
        self.activation = activation
        self.layers = make_layers()

    def call(self, inputs, **kwargs):

        outputs = []

        for layer in self.layers:
            outputs.append(layer(inputs))

        return outputs
