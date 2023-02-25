import tensorflow as tf


class DiversityLoss(tf.keras.losses.Loss):

    def __init__(self, num_vectors):
        super().__init__()
        self.num_vectors = num_vectors

    def call(self, y_true, y_pred):
        return - (self.num_vectors - tf.math.exp(-tf.reduce_sum(y_pred * tf.math.log(y_pred + 10 ** -20), axis=-1))) / \
            self.num_vectors

        # return tf.reduce_mean(y_pred * tf.math.log(y_pred + 10**-20))
