import tensorflow as tf
from typing import List, Tuple
import layers


class Data2VecModel:
    masking: bool
    masking_layer: layers.Masking
    len_latent_space: int
    conv_encoder: layers.ConvFeatureExtractionModel
    transformer_encoder: layers.EncoderTransformer
    top_k_transformer: int
    inputs1: Tuple[int, int, int]
    inputs2: Tuple[int, int, int]

    def __init__(self,
                 masking: bool,
                 masking_layer: layers.Masking,
                 len_latent_space: int,
                 conv_encoder: layers.ConvFeatureExtractionModel,
                 transformer_encoder_s: layers.EncoderTransformer,
                 transformer_encoder_t: layers.EncoderTransformer,
                 top_k_transformer: int,
                 inputs1: Tuple[int, int, int],
                 inputs2: Tuple[int, int, int],
                 ):
        super().__init__()

        self.masking = masking  # bool
        self.masking_layer = masking_layer

        self.len_latent_space = len_latent_space
        self.conv_encoder = conv_encoder
        self.transformer_encoder_s = transformer_encoder_s
        self.transformer_encoder_s._name = 'transformer_encoder_s'
        self.transformer_encoder_t = transformer_encoder_t
        self.transformer_encoder_t._name = 'transformer_encoder_t'
        self.top_k_transformer = top_k_transformer
        self.add = tf.keras.layers.Add()
        self.subtract = tf.keras.layers.Subtract()
        self.mul = tf.keras.layers.Multiply()

        self.inputs1 = tf.keras.layers.Input(inputs1)
        self.inputs2 = tf.keras.layers.Input(inputs2)

    def build(self):
        # self.transformer_encoder_s._name = 'transformer_encoder_s'
        # self.transformer_encoder_t._name = 'transformer_encoder_t'

        latent_student = self.conv_encoder(self.inputs1)

        masked_latent_space = self.masking_layer([latent_student, self.inputs2])

        student_outputs = self.transformer_encoder_s(masked_latent_space,
                                                     training=True,
                                                     top_k_transformer=None)

        latent_teacher = tf.stop_gradient(self.conv_encoder(self.inputs1))

        teacher_outputs = tf.stop_gradient(self.transformer_encoder_t(latent_teacher, training=False,
                                                                      top_k_transformer=self.top_k_transformer))

        outputs = tf.concat([student_outputs, teacher_outputs], axis=0)

        return tf.keras.Model(inputs=[self.inputs1, self.inputs2], outputs=outputs)


if __name__ == '__main__':
    data = tf.random.normal((2, 32, 32))
    conv_layers: List[List[Tuple[int, int, int]]] = [[(64, 3, 1), (64, 3, 1)],
                                                     [(128, 3, 1), (128, 3, 2)],
                                                     [(256, 3, 1), (256, 3, 1)],
                                                     [(512, 3, 1), (512, 3, 2)],
                                                     [(512, 3, 1), (512, 3, 1)],
                                                     [(512, 3, 1), (512, 3, 1)],
                                                     [(512, 3, 1), (512, 3, 1)]]

    num_duplicate_layer: Tuple[int, int, int, int, int, int, int] = (2, 1, 1, 1, 3, 1, 2)
    conv = layers.ConvFeatureExtractionModel(conv_layers=conv_layers, activation='gelu', units=512,
                                             num_duplicate_layer=num_duplicate_layer)

    mask_ = tf.where(tf.random.uniform(shape=(100, 16), maxval=1) > 0.9, 1., 0.)
    inputs = [tf.Variable(tf.random.normal((100, 32, 32, 3))), tf.Variable(mask_)]
    mask = layers.Masking(num_channels=512)
    # mask.build((None, 16))

    encoder = layers.TransformerEncoder(num_layers=24, d_model=512, num_attention_heads=8, dff=4096, dropout_rate=0.1)

    ffn = layers.FFN(dff=512, activation='gelu', num_layers=1)

    model = Data2VecModel(masking=True,
                          masking_layer=mask,
                          len_latent_space=16,
                          conv_encoder=conv,
                          transformer_encoder=encoder,
                          ffn=ffn,
                          tau=0.9,
                          top_k_transformer=4, )

    model.build(([(None, 32, 32, 3), (None, 16)]))
    model.summary()
