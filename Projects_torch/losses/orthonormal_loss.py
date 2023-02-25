
import tensorflow as tf


class OrthonormalLoss(tf.keras.losses.Loss):

    def __init__(self, num_vector):
        super().__init__()
        self.identity = tf.eye(num_vector)

    def call(self, y_true, y_pred):
        y_pred_transpose = tf.transpose(y_pred, perm=[0, 2, 1])
        return tf.norm(y_pred_transpose @ y_pred - self.identity,  ord='fro', axis=[-2, -1])
