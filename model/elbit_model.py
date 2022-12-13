import layers
import tensorflow as tf
from typing import List, Tuple


class TargetDetectorModel:
    input: Tuple[int, int, int]
    conv_encoder: layers.ImageConvFeatureExtractionModel

    def __init__(self,
                 conv_encoder: layers.ImageConvFeatureExtractionModel,
                 input: Tuple[int, int, int]

                 ):
        super().__init__()

        self.input = tf.keras.layers.Input(input)
        self.conv_encoder = conv_encoder
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(unit=1, activation='softmax')

    def build(self):
        outputs = self.conv_encoder(self.input)

        outputs = self.flatten(outputs)

        outputs = self.fc(outputs)

        return tf.keras.Model(inputs=self.input, outputs=outputs)


if __name__ == '__main__':
    conv_layers: List[List[Tuple[int, int, int]]] = [[(64, 3, 1), (64, 3, 1)],
                                                     [(128, 3, 1), (128, 3, 1)],
                                                     [(256, 3, 1), (256, 3, 1)],
                                                     [(512, 3, 1), (512, 3, 1)],
                                                     [(512, 3, 1), (512, 3, 1)],
                                                     [(512, 3, 1), (512, 3, 1)],
                                                     [(512, 3, 1), (512, 3, 1)]]

    num_duplicate_layer: Tuple[int, int, int, int, int, int, int] = (1, 1, 1, 1, 1, 1, 1)
    conv = layers.ImageConvFeatureExtractionModel(conv_layers=conv_layers, activation='gelu', units=512,
                                                  num_duplicate_layer=num_duplicate_layer)

    inputs = tf.Variable(tf.random.normal((100, 32, 32, 3)))

    model = TargetDetectorModel(conv_encoder=conv, input=(1024, 1024, 3))

    model = model.build()

    model.summary()
