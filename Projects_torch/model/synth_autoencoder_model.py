import tensorflow as tf
from typing import List, Tuple
from Projects_tensorflow import layers


class SynthAutoEncoder:
    inputs: Tuple[int, int, int]
    top_k_transformer: int

    # masking_transformer: layers.MaskingTransformer
    conv_encoder: layers.ConvFeatureExtractionModel
    transformer_encoder: layers.EncoderTransformer
    transformer_decoder: layers.DecoderTransformer
    conv_decoder: layers.ConvDecoderModel
    params_predictor: layers.ParamsPredictor
    linear_classifier: layers.LinearClassifier

    def __init__(self,
                 # masking_transformer: layers.MaskingTransformer,
                 conv_encoder: layers.ConvFeatureExtractionModel,
                 transformer_encoder: layers.EncoderTransformer,
                 transformer_decoder: layers.DecoderTransformer,
                 conv_decoder: layers.ConvDecoderModel,
                 params_predictor: layers.ParamsPredictor,
                 linear_classifier: layers.LinearClassifier,
                 top_k_transformer: int,
                 inputs: Tuple[int, int, int],

                 ):
        super().__init__()

        self.inputs = tf.keras.layers.Input(inputs)
        self.top_k_transformer = top_k_transformer

        # self.masking_transformer = masking_transformer

        self.conv_encoder = conv_encoder
        self.transformer_encoder = transformer_encoder

        self.transformer_decoder = transformer_decoder
        self.conv_decoder = conv_decoder

        self.linear_classifier = linear_classifier

        # self.params_predictor = params_predictor

    def build(self):
        inputs = self.inputs
        outputs_conv_encoder = self.conv_encoder(inputs)

        # outputs_conv_encoder = self.masking_transformer(outputs_conv_encoder)
        outputs_conv_encoder_ = tf.keras.layers.Dropout(0.2)(outputs_conv_encoder)

        outputs_transformer_encoder_top_k, outputs_transformer_encoder = self.transformer_encoder(outputs_conv_encoder_,
                                                                                                  training=True,
                                                                                                  top_k_transformer=3)

        outputs_transformer_decoder = self.transformer_decoder(x=outputs_conv_encoder,
                                                               context=outputs_transformer_encoder,
                                                               training=True,
                                                               top_k_transformer=None)

        outputs_wav = self.conv_decoder(outputs_transformer_decoder)

        # outputs_params = self.params_predictor(outputs_transformer_encoder)
        outputs_params_list = self.linear_classifier(outputs_transformer_encoder_top_k)

        wavs = tf.keras.layers.concatenate([inputs, outputs_wav], axis=0, name='wavs')

        latent = tf.keras.layers.concatenate([outputs_conv_encoder, outputs_transformer_decoder],
                                             axis=0, name='latent')

        outputs = [wavs, latent] + [outputs_params_list]

        return tf.keras.Model(inputs=[self.inputs], outputs=outputs)  # [outputs_params, wavs, wavs, latent])


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
