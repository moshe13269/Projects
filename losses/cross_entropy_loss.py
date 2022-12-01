import tensorflow as tf


class CELoss(tf.keras.losses):

    def __init__(self):
        self.ce = tf.keras.losses.CategoricalCrossentropy()

    def call(self, y_true, y_pred):
        return self.ce(y_true, y_pred)
