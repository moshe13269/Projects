import tensorflow as tf
from typing import List, Tuple
from layers.masking import Masking
from layers.ffn import FFN
from layers.transformer_encoder import TransformerEncoder
from layers.conv_image_encoder import ConvFeatureExtractionModel


class Data2VecModel:
    masking: bool
    masking_layer: Masking
    len_latent_space: int
    conv_encoder: ConvFeatureExtractionModel
    transformer_encoder: TransformerEncoder
    ffn: FFN
    tau: float
    top_k_transformer: int
    inputs1: Tuple[int, int, int]
    inputs2: Tuple[int, int, int]

    def __init__(self,
                 masking: bool,
                 masking_layer: Masking,
                 len_latent_space: int,
                 conv_encoder: ConvFeatureExtractionModel,
                 transformer_encoder: TransformerEncoder,
                 ffn: FFN,
                 tau: float,
                 top_k_transformer: int,
                 inputs1: Tuple[int, int, int],
                 inputs2: Tuple[int, int, int],
                 ):

        super().__init__()

        self.masking = masking  # bool
        self.masking_layer = masking_layer

        self.len_latent_space = len_latent_space
        self.conv_encoder = conv_encoder
        self.transformer_encoder = transformer_encoder
        self.ffn = ffn
        self.tau = tau
        self.top_k_transformer = top_k_transformer

        self.inputs1 = tf.keras.layers.Input(inputs1) #(shape=(32, 32, 3,))
        self.inputs2 = tf.keras.layers.Input(inputs2) #(shape=(16,))

    def build(self):
        latent_image = self.conv_encoder(self.inputs1)

        teacher_inputs = tf.stop_gradient(tf.identity(latent_image))

        masked_latent_space = self.masking_layer([latent_image, self.inputs2])

        student_encoding = self.transformer_encoder(masked_latent_space,
                                                    training=True,
                                                    top_k_transformer=1)

        teacher_encoding = self.transformer_encoder(teacher_inputs, training=False,
                                                    top_k_transformer=self.top_k_transformer)

        # teacher_encoding = self.ffn(teacher_encoding)
        # teacher_encoding = tf.reduce_mean(self.ffn(teacher_encoding), axis=1)

        teacher_encoding = teacher_encoding * self.tau + (1 - self.tau) * student_encoding

        teacher_encoding = tf.stop_gradient(teacher_encoding)

        outputs = tf.concat([student_encoding, teacher_encoding], axis=0)

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
    conv = ConvFeatureExtractionModel(conv_layers=conv_layers, activation='gelu', units=512,
                                      num_duplicate_layer=num_duplicate_layer)

    mask_ = tf.where(tf.random.uniform(shape=(100, 16), maxval=1) > 0.9, 1., 0.)
    inputs = [tf.Variable(tf.random.normal((100, 32, 32, 3))), tf.Variable(mask_)]
    mask = Masking(num_channels=512)
    # mask.build((None, 16))

    encoder = TransformerEncoder(num_layers=24, d_model=512, num_attention_heads=8, dff=4096, dropout_rate=0.1)

    ffn = FFN(dff=512, activation='gelu', num_layers=1)

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
