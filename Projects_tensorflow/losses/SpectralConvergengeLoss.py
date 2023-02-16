from abc import ABC
import tensorflow as tf


class SpectralConvergengeLoss(tf.keras.losses.Loss, ABC):

    def __init__(self):
        super(SpectralConvergengeLoss, self).__init__()
        self.loss = tf.keras.losses.MeanSquaredError()
        self.subtract = tf.keras.layers.Subtract()
        self.mul = tf.keras.layers.Multiply()
        self.reshape = tf.keras.layers.Reshape((16384,))

    def call(self, y_true, y_pred):
        x_mag, y_mag = tf.split(y_pred, num_or_size_splits=2, axis=0)

        x_mag = self.reshape(x_mag)
        y_mag = self.reshape(y_mag)
        x_mag = tf.math.abs(tf.signal.stft(x_mag, frame_length=256, frame_step=128, fft_length=256))
        y_mag = tf.math.abs(tf.signal.stft(y_mag, frame_length=256, frame_step=128, fft_length=256))

        x = tf.norm(self.subtract([y_mag, x_mag]), ord='fro', axis=[-2, -1]) / \
            (tf.norm(y_mag, ord='fro', axis=[-2, -1]) + 10**-10)
        return x # tf.reduce_mean(self.mul([x, x]))
