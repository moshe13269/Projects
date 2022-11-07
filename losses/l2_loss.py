from abc import ABC
import tensorflow as tf


class L2Loss(tf.keras.losses.Loss, ABC):

    def __init__(self, tile_params):
        super(L2Loss, self).__init__()
        self.loss = tf.keras.losses.MeanSquaredError()
        self.tile_params = tf.constant(tile_params)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, y_true, y_pred):
        teacher_encoding, student_encoding = tf.split(y_pred, num_or_size_splits=2, axis=0)

        teacher_encoding = self.flatten(teacher_encoding)
        student_encoding = self.flatten(student_encoding)
        y_true = tf.tile(y_true, self.tile_params)

        teacher_encoding = tf.boolean_mask(teacher_encoding, mask=y_true)
        student_encoding = tf.boolean_mask(student_encoding, mask=y_true)

        return self.loss(teacher_encoding, student_encoding)