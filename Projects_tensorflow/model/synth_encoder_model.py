import tensorflow as tf
from typing import List, Tuple
from Projects_tensorflow import layers


class SynthEncoder:
    inputs: Tuple[int, int, int]
    top_k_transformer: int

    conv_encoder: layers.ConvFeatureExtractionModel
    transformer_encoder: layers.EncoderTransformer
    transformer_decoder: layers.DecoderTransformer
    conv_decoder: layers.ConvDecoderModel
    params_predictor: layers.ParamsPredictor
    linear_classifier: layers.LinearClassifier
    # masking_transformer: layers.MaskingTransformer

    def __init__(self,
                 conv_encoder: layers.ConvFeatureExtractionModel,
                 transformer_encoder: layers.EncoderTransformer,
                 linear_classifier: layers.LinearClassifier,
                 # masking_transformer: layers.MaskingTransformer,
                 top_k_transformer: int,
                 inputs: Tuple[int, int, int],

                 ):
        super().__init__()

        self.inputs = tf.keras.layers.Input(inputs)
        self.top_k_transformer = top_k_transformer

        self.conv_encoder = conv_encoder
        self.transformer_encoder = transformer_encoder
        # self.masking_transformer = masking_transformer

        self.linear_classifier = linear_classifier

    def build(self):
        inputs = self.inputs
        outputs_conv_encoder = self.conv_encoder(inputs)

        # outputs_conv_encoder = self.masking_transformer(outputs_conv_encoder)
        outputs_conv_encoder = tf.keras.layers.Dropout(0.35)(outputs_conv_encoder)

        outputs_transformer_encoder = self.transformer_encoder(outputs_conv_encoder,
                                                               training=True,
                                                               top_k_transformer=3)

        outputs = self.linear_classifier(outputs_transformer_encoder)

        return tf.keras.Model(inputs=[self.inputs], outputs=outputs)


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
