from abc import ABC
import tensorflow as tf


class CosSimLoss(tf.keras.losses.Loss, ABC):

    def __init__(self):
        super(CosSimLoss, self).__init__()
        self.loss = tf.keras.losses.CosineSimilarity(axis=-1)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, y_true, y_pred):
        y_pred, y_true = tf.split(y_pred, num_or_size_splits=2, axis=0)
        # tf.print(tf.shape(y_pred), tf.shape(y_true))
        return - self.loss(self.flatten(y_true), self.flatten(y_pred))
