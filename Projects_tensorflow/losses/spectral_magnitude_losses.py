from abc import ABC
import tensorflow as tf


class Spectral_LogSTFTMagnitude_Loss(tf.keras.losses.Loss, ABC):

    def __init__(self, stft=True):
        super(Spectral_LogSTFTMagnitude_Loss, self).__init__()
        self.loss = tf.keras.losses.MeanSquaredError()
        self.subtract = tf.keras.layers.Subtract()
        self.mul = tf.keras.layers.Multiply()
        self.reshape = tf.keras.layers.Reshape((16384,))
        self.mae_loss = tf.keras.losses.MeanAbsoluteError()
        self.stft = stft

    def call(self, y_true, y_pred):
        x_mag, y_mag = tf.split(y_pred, num_or_size_splits=2, axis=0)

        if not self.stft:
            x_mag = tf.math.abs(
                tf.signal.stft(self.reshape(x_mag), frame_length=256, frame_step=128, fft_length=256))

            y_mag = tf.math.abs(
                tf.signal.stft(self.reshape(y_mag), frame_length=256, frame_step=128, fft_length=256))

        x_mag_log = tf.math.log(x_mag + 1.) #x_mag #
        y_mag_log = tf.math.log(y_mag + 1.) #y_mag #

        log_stft_magnitude_loss = self.mae_loss(x_mag_log, y_mag_log)

        spectral_convergent_loss = tf.norm(self.subtract([y_mag, x_mag]), ord='fro', axis=[-2, -1]) / \
            (tf.norm(y_mag, ord='fro', axis=[-2, -1]) + 10**-10)
        return log_stft_magnitude_loss + spectral_convergent_loss
