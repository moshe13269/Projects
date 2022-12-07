import tensorflow as tf
from typing import List, Tuple
import layers


class SynthEncoder:
    inputs: Tuple[int, int, int]
    top_k_transformer: int

    conv_encoder: layers.ConvFeatureExtractionModel
    transformer_encoder: layers.EncoderTransformer
    transformer_decoder: layers.DecoderTransformer
    conv_decoder: layers.ConvDecoderModel

    def __init__(self,
                 conv_encoder: layers.ConvFeatureExtractionModel,
                 transformer_encoder: layers.EncoderTransformer,
                 transformer_decoder: layers.DecoderTransformer,
                 conv_decoder: layers.ConvDecoderModel,
                 top_k_transformer: int,
                 inputs: Tuple[int, int, int],

                 ):
        super().__init__()

        self.inputs = tf.keras.layers.Input(inputs)
        self.top_k_transformer = top_k_transformer

        self.conv_encoder = conv_encoder
        self.transformer_encoder = transformer_encoder

        self.transformer_decoder = transformer_decoder
        self.conv_decoder = conv_decoder
        # self.linear_classifier = linear_classifier

        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.dropout2 = tf.keras.layers.Dropout(0.1)
        self.fc1 = tf.keras.layers.Dense(1, activation='relu')
        self.fc2 = tf.keras.layers.Dense(16, activation='relu')
        self.fc3 = tf.keras.layers.Dense(16, activation='sigmoid')

        self.reshape = tf.keras.layers.Reshape(target_shape=(50,))  # target_shape=(338,)
        # self.permute = tf.keras.layers.Permute((1, 2))
        # self.activation = tf.keras.layers.Activation('relu')

    def build(self):
        outputs_conv_encoder = self.conv_encoder(self.inputs)

        outputs_transformer_encoder = self.transformer_encoder(outputs_conv_encoder,
                                                               training=True,
                                                               top_k_transformer=3)

        outputs_transformer_decoder = self.transformer_encoder(x=outputs_conv_encoder,
                                                               context=outputs_transformer_encoder,
                                                               training=True,
                                                               top_k_transformer=3)

        outputs_wav = self.conv_decoder(outputs_transformer_decoder)

        # outputs = self.permute(outputs)
        outputs = self.dropout1(outputs_transformer_encoder)
        outputs = self.fc1(outputs)
        outputs = self.reshape(outputs)
        outputs = self.dropout2(self.fc2(outputs))
        outputs_params = self.fc3(outputs)  # tf.keras.activations.sigmoid(self.fc3(outputs))

        return tf.keras.Model(inputs=[self.inputs], outputs=[outputs_params, outputs_conv_encoder,
                                                             # outputs_transformer_encoder,
                                                             outputs_transformer_decoder,
                                                             outputs_wav])


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
