import tensorflow as tf
from typing import List, Tuple
import layers


class SynthEncoder:
    transformer_encoder: layers.EncoderTransformer
    inputs: Tuple[int, int, int]
    top_k_transformer: int
    conv_encoder: layers.ConvFeatureExtractionModel

    def __init__(self,
                 conv_encoder: layers.ConvFeatureExtractionModel,
                 transformer_encoder: layers.EncoderTransformer,
                 top_k_transformer: int,
                 inputs: Tuple[int, int, int],
                 ):
        super().__init__()

        self.inputs = tf.keras.layers.Input(inputs)
        self.top_k_transformer = top_k_transformer

        self.conv_encoder = conv_encoder
        self.transformer_encoder = transformer_encoder

        self.dropout = tf.keras.layers.Dropout(0.1)

        self.fc1 = tf.keras.layers.Dense(1, activation='relu')
        self.fc2 = tf.keras.layers.Dense(16, activation='relu')
        self.fc3 = tf.keras.layers.Dense(16, activation='sigmoid')

        self.reshape = tf.keras.layers.Reshape(target_shape=(338,))
        self.permute = tf.keras.layers.Permute((1, 2))

    def build(self):
        outputs = self.conv_encoder(self.inputs)

        outputs = self.transformer_encoder(outputs,
                                           training=True,
                                           top_k_transformer=self.top_k_transformer)

        outputs = self.permute(outputs)
        # outputs = self.dropout(outputs)
        outputs = self.dropout(self.fc1(outputs))
        outputs = self.reshape(outputs)
        outputs = self.fc2(outputs)
        # outputs = self.fc3(outputs)

        return tf.keras.Model(inputs=[self.inputs], outputs=outputs)


if __name__ == '__main__':
    data = tf.random.normal((2, 32, 32))
    conv_layers: List[Tuple[int, int, int]] = [(512, 10, 5),
                                               (512, 3, 2),
                                               (512, 3, 2),
                                               (512, 3, 2),
                                               (512, 2, 2),
                                               (512, 2, 2),
                                               (512, 2, 2)]

    num_duplicate_layer: Tuple[int, int, int, int, int, int, int] = (1, 1, 1, 1, 1, 1, 1)
    conv = layers.ConvFeatureExtractionModel(conv_layers=conv_layers, activation='gelu', units=512,
                                             num_duplicate_layer=num_duplicate_layer)

    inputs = tf.random.normal((2, 16384, 1))

    encoder = layers.EncoderTransformer(num_layers=24, d_model=512, num_attention_heads=8, dff=4096, dropout_rate=0.1,
                                        activation='relu')

    inputs_shape: Tuple[int, int] = (16384, 1)

    model = SynthEncoder(conv_encoder=conv,
                         transformer_encoder=encoder,
                         inputs=inputs_shape)

    model.build()
    model.summary()

