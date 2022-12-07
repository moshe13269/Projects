import tensorflow as tf


class CELoss(tf.keras.losses.Loss):

    def __init__(self):
        super(CELoss, self).__init__()
        self.ce = tf.keras.losses.CategoricalCrossentropy()

    def call(self, y_true, y_pred):
        # tf.print(tf.shape(y_pred), tf.shape(y_true))
        return self.ce(y_true, y_pred)
