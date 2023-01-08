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
                 # num_classes_per_param: List[int],
                 activation: str = 'softmax',
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)

        # self.num_classes_per_param = num_classes_per_param
        # self.indexes = [sum(self.num_classes_per_param[:i])
        #                 for i in range(len(self.num_classes_per_param)+1)]

        def make_layers():
            layers = []
            for output_dim in self.outputs_dimension_per_outputs:
                layers.append(
                    tf.keras.Sequential([
                        # ReLU(),
                        # Dropout(rate=dropout),
                        # tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1)),
                        Flatten(),
                        Dense(units=output_dim, activation='relu'),
                        Dropout(rate=dropout),
                        Dense(units=output_dim, activation='relu'),
                        Dropout(rate=dropout),
                        Dense(units=output_dim, activation=None),
                    ])
                )
                # layers.append(Dense(units=output_dim, activation=self.activation))
            return layers

        self.outputs_dimension_per_outputs = outputs_dimension_per_outputs
        self.activation = activation
        self.layers = tf.keras.Sequential([
            Flatten(),
            Dense(units=512 * 50, activation='gelu'),
            Dropout(rate=dropout),
            Dense(units=sum(outputs_dimension_per_outputs), activation='gelu'),
            Dropout(rate=dropout),
            Dense(units=sum(outputs_dimension_per_outputs), activation=None),
        ])

    def call(self, inputs, **kwargs):
        outputs = self.layers(inputs)

        return outputs
