import tensorflow as tf
from typing import List, Tuple
from layers.masking import Masking
from layers.ffn import FFN
from layers.transformer_encoder import TransformerEncoder
from layers.conv_image_encoder import ConvFeatureExtractionModel
from tensorflow.python.keras import Input
from dataclasses import dataclass, field


class Data2VecModel(tf.keras.Model):
    masking: bool
    masking_layer: Masking
    len_latent_space: int
    conv_encoder: ConvFeatureExtractionModel
    transformer_encoder: TransformerEncoder
    ffn: FFN
    tau: float
    top_k_transformer: int

    def __init__(self,
                 masking: bool,
                 masking_layer: Masking,
                 len_latent_space: int,
                 conv_encoder: ConvFeatureExtractionModel,
                 transformer_encoder: TransformerEncoder,
                 ffn: FFN,
                 tau: float,
                 top_k_transformer: int
                 ):

        super(Data2VecModel, self).__init__()

        # self.prob2mask = prob2mask
        # self.masking_length = masking_length

        self.masking = masking  # bool
        self.masking_layer = masking_layer

        self.len_latent_space = len_latent_space
        self.conv_encoder = conv_encoder
        self.transformer_encoder = transformer_encoder
        self.ffn = ffn
        self.tau = tau
        self.top_k_transformer = top_k_transformer

    # def build(self, input_shape=([(None, 32, 32, 3), (None, 16)])):
    #     inputs = Input(shape=([(None, 32, 32, 3), (None, 16)]))
    # def __call__(self, inputs, **kwargs):

    def call(self, inputs, **kwargs):
        # input = tf.keras.layers.InputLayer([(None, 32, 32, 3), (None, 16)])

        image_file = inputs[0]
        mask = inputs[1]

        latent_image = self.conv_encoder(image_file)

        teacher_inputs = tf.stop_gradient(tf.identity(latent_image))

        if self.masking:
            masked_latent_space, mask = self.masking_layer([latent_image, mask])
            # tf.print(tf.reduce_mean(masked_latent_space), tf.reduce_mean(mask))
        else:
            masked_latent_space = latent_image
            mask = tf.Variable([0])

        student_encoding = self.transformer_encoder(masked_latent_space,
                                                    training=True,
                                                    top_k_transformer=1)[0]

        teacher_encoding = tf.stop_gradient(self.transformer_encoder(teacher_inputs,
                                                                     training=False,
                                                                     top_k_transformer=self.top_k_transformer))

        teacher_encoding = tf.stop_gradient(tf.reduce_mean(self.ffn(teacher_encoding), axis=1))

        teacher_encoding = tf.stop_gradient(teacher_encoding * self.tau + (1 - self.tau) * student_encoding)

        return tf.concat([teacher_encoding, student_encoding], axis=1), mask

    # @self.input_shape.setter
    # def input_shape(self, value):
    #     self._input_shape = value


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
    # input1 = Input(shape=(32, 32, 3,))
    # input2 = Input(shape=(4,))
    # input = tf.keras.layers.Concatenate()([input1, input2])
    # inputs = tf.keras.Input((, ))
    # model = tf.keras.Model(inputs=inputs, outputs=model(inputs))
    # input = Input(shape=(32, 32, 3,))
    # model = Model(inputs=Input(shape=(32, 32, 3,)), outputs=Data2VecModel()(Input(shape=(32, 32, 3,)))
    # print(model.summary(expand_nested=True))
    # model.build(input_shape=([(None, 32, 32, 3), (None, 16)]))
    model.build(([(None, 32, 32, 3), (None, 16)]))
    model.summary()
    a=0
    # outputs = model(inputs)
    # print(tf.reduce_mean(outputs[0]))
    # self.feature_extractor = ConvFeatureExtractionModel(conv_layers=cfg)
    # # conv_layers: List[Tuple[int, int, int]] = [(512, 10, 5), (512, 5, 3), (512, 3, 2)]
