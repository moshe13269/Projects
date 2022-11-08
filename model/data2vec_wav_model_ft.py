import tensorflow as tf
from typing import List, Tuple
from layers.masking import Masking
from layers.ffn import FFN
from model.data2vec_wav_model import Data2VecModel


class Data2VecModelFT:
    top_k_transformer: int
    inputs: Tuple[int, int, int]
    path2load_model: str

    def __init__(self,
                 top_k_transformer: int,
                 inputs: Tuple[int, int, int],
                 path2load_model: str
                 ):

        super().__init__()
        self.model = tf.keras.models.load_model(path2load_model, compile=False)
        self.model.trainable = False
        # self.model.tr
        self.conv_encoder = self.model.layers[1]
        self.conv_encoder.trainable = False
        self.transformer_encoder = self.model.layers[6]
        self.transformer_encoder.trainable = False
        self.top_k_transformer = top_k_transformer
        self.inputs = tf.keras.layers.Input(inputs)
        self.fc1 = tf.keras.layers.Dense(1)
        self.fc2 = tf.keras.layers.Dense(16)
        self.flatten = tf.keras.layers.Flatten()

    def build(self):
        latent_space = self.conv_encoder(self.inputs)

        outputs = self.transformer_encoder(latent_space,
                                           training=False,
                                           top_k_transformer=1)

        outputs = tf.keras.layers.Activation('relu')(outputs)
        # outputs = self.fc1(outputs)
        # outputs = tf.keras.layers.Activation('relu')(outputs)

        outputs = self.flatten(outputs)

        outputs = self.fc2(outputs)
        outputs = tf.keras.layers.Activation('relu')(outputs)
        return tf.keras.Model(inputs=self.inputs, outputs=outputs)


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
