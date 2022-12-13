from abc import ABC
import tensorflow as tf


class LogSTFTMagnitudeLoss(tf.keras.losses.Loss, ABC):

    def __init__(self):
        super(LogSTFTMagnitudeLoss, self).__init__()
        self.loss = tf.keras.losses.MeanSquaredError()
        self.subtract = tf.keras.layers.Subtract()
        self.mul = tf.keras.layers.Multiply()

    def call(self, y_true, y_pred):
        x_mag, y_mag = tf.split(y_pred, num_or_size_splits=2, axis=0)
        x_mag = tf.math.log(tf.math.abs(
            tf.signal.stft(tf.squeeze(x_mag), frame_length=256, frame_step=128, fft_length=256)))
        y_mag = tf.math.log(tf.math.abs(
            tf.signal.stft(tf.squeeze(y_mag), frame_length=256, frame_step=128, fft_length=256)))

        return tf.reduce_mean(tf.math.abs(self.subtract([y_mag, x_mag])))
