from abc import ABC
import tensorflow as tf


class L2Loss(tf.keras.losses.Loss, ABC):

    def __init__(self, tile_params):
        super(L2Loss, self).__init__()
        self.loss = tf.keras.losses.MeanSquaredError()

    def call(self, y_true, y_pred):
        teacher_encoding, student_encoding = tf.split(y_pred, num_or_size_splits=2, axis=0)
        # a = tf.reduce_sum(tf.where(tf.expand_dims(y_true, axis=2) == 1.0, 1., 0.)) * 512
        # b = tf.reduce_sum(tf.where(tf.expand_dims(y_true, axis=2) == 0.0, 1., 0.)) * 512
        return self.loss(teacher_encoding * tf.expand_dims(y_true, axis=2), student_encoding * tf.expand_dims(y_true, axis=2))
        # res = (teacher_encoding * tf.expand_dims(y_true, axis=2) - student_encoding * tf.expand_dims(y_true, axis=2))**2
        # return res #/a

        # return self.loss(teacher_encoding * tf.expand_dims(y_true, axis=2), student_encoding * tf.expand_dims(y_true, axis=2)) * (50/16)



        # y_true = self.reshape(y_true)
        # teacher_encoding = self.flatten(teacher_encoding)
        # student_encoding = self.flatten(student_encoding)
        # tf.print(tf.shape(teacher_encoding), tf.shape(student_encoding), tf.shape(y_true))
        # tf.print(tf.reduce_sum(y_true))
        # teacher_encoding = tf.ragged.boolean_mask(teacher_encoding, mask=tf.cast(y_true, tf.bool))
        # teacher_encoding = tf.boolean_mask(teacher_encoding, mask=y_true, axis=0)
        # student_encoding = tf.boolean_mask(student_encoding, mask=y_true, axis=0)
        # tf.print(tf.shape(teacher_encoding), tf.shape(student_encoding), tf.shape(y_true))

        # return tf.reduce_mean(teacher_encoding) - tf.reduce_mean(student_encoding)


        # return self.loss(teacher_encoding, student_encoding)

