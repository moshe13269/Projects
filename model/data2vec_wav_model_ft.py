import tensorflow as tf
from typing import List, Tuple
from layers.masking import Masking
from layers.ffn import FFN
from model.data2vec_wav_model import Data2VecModel


class Data2VecModelFT:
    top_k_transformer: int
    inputs: Tuple[int, int] #, int]
    path2load_model: str
    model_trainable: bool

    def __init__(self,
                 top_k_transformer: int,
                 inputs: Tuple[int, int],#, int],
                 path2load_model: str,
                 model_trainable: bool
                 ):

        super().__init__()
        self.model = tf.keras.models.load_model(path2load_model, compile=False)
        self.model.trainable = model_trainable
        self.conv_encoder = self.model.layers[1]
        self.conv_encoder.trainable = model_trainable
        self.transformer_encoder = self.model.layers[5]
        self.transformer_encoder.trainable = model_trainable
        self.top_k_transformer = top_k_transformer
        self.inputs = tf.keras.layers.Input(inputs)
        self.fc1 = tf.keras.layers.Dense(1, activation='relu')
        self.fc2 = tf.keras.layers.Dense(16, activation='relu')
        self.fc3 = tf.keras.layers.Dense(16, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.top_k_transformer = top_k_transformer

    def build(self):
        latent_space = self.conv_encoder(self.inputs)

        outputs = self.transformer_encoder(latent_space,
                                           training=False,
                                           top_k_transformer=None)

        # outputs = tf.keras.layers.Activation('relu')(outputs)

        outputs = self.fc1(outputs)
        outputs = self.fc2(tf.squeeze(outputs, axis=-1))
        outputs = self.fc3(outputs)

        # outputs = tf.keras.layers.Activation('sigmoid')(outputs)
        return tf.keras.Model(inputs=self.inputs, outputs=outputs)


if __name__ == '__main__':
    data = tf.random.normal((2, 16384, 1))
    top_k_transformer: int
    inputs: Tuple[int, int]
    path2load_model: str
    model_trainable: bool

    path2model = '/home/moshelaufer/PycharmProjects/results/checkpoint/data2vec_wav_1/'

    model = Data2VecModelFT(top_k_transformer=1, inputs=(16384, 1), model_trainable=False, path2load_model=path2model)

    model = model.build()

    a = model(data)

    print(a.shape)
    print(a)
