
import tensorflow as tf


class CustomAccuracy(tf.keras.metrics.Metric):

    def __init__(self,
                 outputs_dimension_per_outputs,
                 num_classes,
                 index_y_true,
                 name='Accuracy',
                 **kwargs):
        super(CustomAccuracy, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.index_y_true = index_y_true

        self.outputs_dimension_per_outputs = outputs_dimension_per_outputs
        self.index2split = [sum(self.outputs_dimension_per_outputs[:i])
                            for i in range(len(self.outputs_dimension_per_outputs) + 1)]

        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.acc = tf.keras.metrics.Accuracy()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.squeeze(self.convert_matrix2one_hot(y_true), axis=1)
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        self.acc.update_state(y_true, y_pred)

    def result(self):
        return self.acc.result()
        # return self.true_positives

    def set_indexes(self):
        self.indexes = [i for i in range(self.num_classes)]

    def reset_states(self):
        self.acc.reset_states()
        # self.true_positives.assign(0)

    @tf.autograph.experimental.do_not_convert
    def split(self, inputs):
        return [inputs[:, self.index2split[i]: self.index2split[i + 1]] for i in range(len(self.index2split) - 1)]