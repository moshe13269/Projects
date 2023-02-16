import tensorflow as tf
from typing import List, Tuple
import layers


class SynthAutoEncoder:
    inputs: Tuple[int, int, int]
    top_k_transformer: int

    conv_encoder: layers.ConvFeatureExtractionModel
    conv_decoder: layers.ConvDecoderModel

    transformer: layers.Transformer
    linear_classifier: layers.LinearClassifier

    def __init__(self,
                 conv_encoder: layers.ConvFeatureExtractionModel,
                 conv_decoder: layers.ConvDecoderModel,

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

        self.conv_encoder = conv_encoder
        self.conv_decoder = conv_decoder

        self.transformer = transformer

        self.linear_classifier = linear_classifier

    def build(self):

        inputs1 = self.inputs1
        inputs2 = self.inputs2
        inputs3 = self.inputs3

        outputs_conv_encoder = self.conv_encoder(inputs1)

        decoder_outputs, encoder_outputs = self.transformer([outputs_conv_encoder, inputs2, inputs3])

        outputs_wav = self.conv_decoder(decoder_outputs)

        outputs_params_list = self.linear_classifier(encoder_outputs)

        wavs = tf.keras.layers.concatenate([inputs1, outputs_wav], axis=0, name='wavs')

        # latent = tf.keras.layers.concatenate([outputs_conv_encoder, decoder_outputs],
        #                                      axis=0, name='latent')

        # outputs = [wavs] + outputs_params_list #[wavs, latent] + outputs_params_list
        outputs = [wavs, outputs_params_list]

        return tf.keras.Model(inputs=[self.inputs1, self.inputs2, self.inputs3], outputs=outputs)


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
