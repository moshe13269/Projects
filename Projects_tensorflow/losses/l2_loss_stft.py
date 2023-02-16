from abc import ABC
import tensorflow as tf


class L2LossSTFT(tf.keras.losses.Loss, ABC):

    def __init__(self):
        super(L2LossSTFT, self).__init__()
        self.loss = tf.keras.losses.MeanSquaredError()
        self.subtract = tf.keras.layers.Subtract()
        self.mul = tf.keras.layers.Multiply()

    def call(self, y_true, y_pred):
        student, teacher = tf.split(y_pred, num_or_size_splits=2, axis=0)
        student = tf.math.abs(tf.signal.stft(tf.squeeze(student), frame_length=256, frame_step=128, fft_length=256))
        teacher = tf.math.abs(tf.signal.stft(tf.squeeze(teacher), frame_length=256, frame_step=128, fft_length=256))
        x = self.subtract([student, teacher])
        return tf.reduce_mean(self.mul([x, x]))
