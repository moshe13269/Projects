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
    dropout: float = 0.2

    def __init__(self,
                 outputs_dimension_per_outputs,
                 # num_classes_per_param: List[int],
                 activation: str = 'relu',
                 dropout: float = 0.2,
                 **kwargs):
        super().__init__(**kwargs)

        self.outputs_dimension_per_outputs = outputs_dimension_per_outputs
        self.activation = activation
        self.layers = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling1D(), #Flatten(),
            Dense(units=sum(outputs_dimension_per_outputs), activation=self.activation), #Dense(units=512 * 50, activation=self.activation),
            Dropout(rate=dropout),
            # Dense(units=sum(outputs_dimension_per_outputs), activation=self.activation),
            # Dropout(rate=dropout),
            Dense(units=sum(outputs_dimension_per_outputs), activation=self.activation),
            Dropout(rate=dropout),
            Dense(units=sum(outputs_dimension_per_outputs), activation=self.activation),
            Dropout(rate=dropout),
            Dense(units=sum(outputs_dimension_per_outputs), activation=None)
        ])

    def call(self, inputs, **kwargs):
        outputs = self.layers(inputs)

        return outputs
