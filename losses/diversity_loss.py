
import tensorflow as tf
from tensorflow.keras.losses import Loss


class DiversityLoss(Loss):

    def __init__(self, temp=1.):
        super().__init__()
        self.temp = temp

    def call(self, y_true, y_pred):
        prob = tf.nn.softmax(y_pred, axis=-1)
        _, _, G, V = y_pred.shape
        return (G * V - tf.reduce_sum(tf.math.exp(- tf.reduce_sum(prob, axis=-1)), axis=-1)) / (G * V)


if __name__ == '__main__':
    loss = DiversityLoss()
    y_pred = tf.random.normal(shape=(10, 40, 2, 16))
    res = loss(y_pred, y_pred)
    print(res.shape)
    print(res)