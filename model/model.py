import tensorflow as tf
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation


class Encoder(Model):
    model: Model

    def __init__(self):
        super().__init__()

    def __build__(self):
        input = Input(shape=(80, 8))

        # self.activation = Activation('gelu')
        #
        # self.cn1 = Conv1D(128, kernel_size=6, strides=3, padding='valid')
        # self.ln1 = BatchNormalization(axis=1)
        #
        # self.cn2 = Conv1D(128, kernel_size=3, strides=2, padding='valid')
        # self.ln2 = BatchNormalization(axis=1)
        #
        # self.cn3 = Conv1D(128, kernel_size=2, strides=1, padding='valid')
        # self.ln3 = BatchNormalization(axis=1)
        #
        # self.cn4 = Conv1D(128, kernel_size=2, strides=1, padding='valid')
        #

        x = Conv1D(128, kernel_size=6, strides=3, padding='valid')(input)
        x = BatchNormalization(axis=1)(x)
        x = Activation('gelu')(x)

        x = Conv1D(128, kernel_size=3, strides=2, padding='valid')(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation('gelu')(x)

        x = Conv1D(128, kernel_size=2, strides=1, padding='valid')(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation('gelu')(x)

        x = Conv1D(128, kernel_size=2, strides=1, padding='valid')(x)

        Model(inputs=[(80, 8)], outputs=[x])


    # def call(self, inputs, **kwargs):
    #     x = self.activation(self.ln1(self.cn1(inputs)))
    #     x = self.activation(self.ln2(self.cn2(x)))
    #     x = self.activation(self.ln3(self.cn3(x)))
    #     x = self.cn4(x)
    #     return x


model = Encoder()
tf.keras.utils.plot_model(model, to_file=r'C:\Users\moshel\Desktop\model.png', show_shapes=True)


input = tf.keras.Input(shape=(80, 8), dtype='float', name='input')
x = tf.keras.layers.Embedding(
    output_dim=512, input_dim=10000, input_length=100)(input)
x = Conv1D(128, kernel_size=6, strides=3, padding='valid')(input)
x = BatchNormalization(axis=1)(x)
x = Activation('gelu')(x)

x = Conv1D(128, kernel_size=3, strides=2, padding='valid')(x)
x = BatchNormalization(axis=1)(x)
x = Activation('gelu')(x)

x = Conv1D(128, kernel_size=2, strides=1, padding='valid')(x)
x = BatchNormalization(axis=1)(x)
x = Activation('gelu')(x)

x = Conv1D(128, kernel_size=2, strides=1, padding='valid')(x)
# output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)
model = tf.keras.Model(inputs=[input], outputs=[x])
dot_img_file = '/tmp/model_1.png'
tf.keras.utils.plot_model(model, to_file=r'C:\Users\moshel\Desktop\model.png', show_shapes=True)