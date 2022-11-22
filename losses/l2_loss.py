from abc import ABC
import tensorflow as tf


class L2Loss(tf.keras.losses.Loss, ABC):

    def __init__(self, tile_params):
        super(L2Loss, self).__init__()
        self.loss = tf.keras.losses.MeanSquaredError()

    def call(self, y_true, y_pred):
        student_encoding, teacher_encoding = tf.split(y_pred, num_or_size_splits=2, axis=0)

        return self.loss(teacher_encoding * tf.expand_dims(y_true, axis=2),
                         student_encoding * tf.expand_dims(y_true, axis=2))
