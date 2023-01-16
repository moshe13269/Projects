
import tensorflow as tf


class CustomAccuracy(tf.keras.metrics.Metric):

    def __init__(self,
                 num_classes,
                 index_y_true,
                 name='AccuracyCustum',
                 **kwargs):
        super(CustomAccuracy, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.index_y_true = index_y_true
        self.indexes = [i for i in range(self.num_classes)]

        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.acc = tf.keras.metrics.Accuracy()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.squeeze(self.convert_matrix2one_hot(y_true), axis=1)
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        self.acc.update_state(y_true, y_pred)
        # values = tf.where(y_true == y_pred, 1., 0.)
        # self.acc.update_state(y_true, y_pred)

        # y_true = tf.cast(y_true, tf.bool)
        # y_pred = tf.cast(y_pred, tf.bool)
        #
        # values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        # values = tf.cast(values, self.dtype)
        # if sample_weight is not None:
        #     sample_weight = tf.cast(sample_weight, self.dtype)
        #     sample_weight = tf.broadcast_to(sample_weight, values.shape)
        #     values = tf.multiply(values, sample_weight)
        # self.true_positives.assign_add(tf.reduce_sum(values))
        # tf.print(self.true_positives)

    @tf.autograph.experimental.do_not_convert
    def convert_matrix2one_hot(self, y_true):
        y_true = tf.split(y_true, num_or_size_splits=9, axis=1)[self.index_y_true]  # (b, 16, 17) -> (b, 1, 17)
        y_true = tf.squeeze(y_true, axis=1)  # (b, 1, 17) -> (b, 17)
        y_true = tf.gather(y_true, indices=[self.indexes], axis=1)
        return tf.cast(y_true, dtype=tf.int32)

    def result(self):
        return self.acc.result()
        # return self.true_positives

    def set_indexes(self):
        self.indexes = [i for i in range(self.num_classes)]

    def reset_states(self):
        self.acc.reset_states()
        # self.true_positives.assign(0)
