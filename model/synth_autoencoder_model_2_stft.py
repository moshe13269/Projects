import tensorflow as tf
from typing import List, Tuple
import layers


class SynthAutoEncoder:
    inputs: Tuple[int, int, int]
    top_k_transformer: int

    transformer: layers.Transformer
    linear_classifier: layers.LinearClassifier

    def __init__(self,

                 transformer: layers.Transformer,

                 linear_classifier: layers.LinearClassifier,
                 inputs1: Tuple[int, int, int],
                 inputs2: Tuple[int, int, int],
                 inputs3: Tuple[int, int, int],
                 ):
        super().__init__()

        self.inputs1 = tf.keras.layers.Input(inputs1)
        self.inputs2 = tf.keras.layers.Input(inputs2)
        self.inputs3 = tf.keras.layers.Input(inputs3)

        self.transformer = transformer

        self.linear_classifier = linear_classifier

        self.fc = tf.keras.layers.Dense(inputs1, activation='tanh')

    def build(self):

        inputs1 = self.inputs1
        inputs2 = self.inputs2
        inputs3 = self.inputs3

        decoder_outputs, encoder_outputs = self.transformer([inputs1, inputs2, inputs3])

        outputs_params_list = self.linear_classifier(encoder_outputs)

        outputs_wav = self.fc(decoder_outputs)

        wavs = tf.keras.layers.concatenate([inputs1, outputs_wav], axis=0, name='wavs')

        return tf.keras.Model(inputs=[self.inputs1, self.inputs2, self.inputs3],
                              outputs=[wavs, outputs_params_list])


if __name__ == '__main__':
    data = tf.random.normal((2, 32, 32))
    conv_layers: List[Tuple[int, int, int]] = [(512, 10, 5),
                                               (512, 3, 2),
                                               (512, 3, 2),
                                               (512, 3, 2),
                                               (512, 3, 2),
                                               (512, 2, 2),
                                               (512, 2, 2)]

    num_duplicate_layer: Tuple[int, int, int, int, int, int, int] = (1, 1, 1, 1, 1, 1, 1)
    conv = layers.ConvFeatureExtractionModel(conv_layers=conv_layers, activation='gelu', units=512,
                                             num_duplicate_layer=num_duplicate_layer)

    inputs = tf.random.normal((2, 16384, 1))

    encoder = layers.EncoderTransformer(num_layers=8, d_model=512, num_attention_heads=8, dff=768, dropout_rate=0.1,
                                        activation='relu')

    inputs_shape: Tuple[int, int] = (16384, 1)

    model = SynthEncoder(conv_encoder=conv,
                         transformer_encoder=encoder,
                         inputs=inputs_shape,
                         top_k_transformer=2)

    m = model.build()
    # outputs = m(inputs)
    # print(outputs.shape)
    m.summary()
