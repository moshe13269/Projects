
import tensorflow as tf


class CustomAccuracy2(tf.keras.metrics.Metric):

    def __init__(self,
                 outputs_dimension_per_outputs,
                 name='Accuracy',
                 **kwargs):
        super(CustomAccuracy2, self).__init__(name=name, **kwargs)

        self.outputs_dimension_per_outputs = outputs_dimension_per_outputs
        self.index2split = [sum(self.outputs_dimension_per_outputs[:i])
                            for i in range(len(self.outputs_dimension_per_outputs) + 1)]
        self.acc = [tf.keras.metrics.Accuracy()
                    for i in range(len(self.outputs_dimension_per_outputs))]
        # self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def set_acc_classes(self, outputs_dimension_per_outputs):
        self.outputs_dimension_per_outputs = outputs_dimension_per_outputs
        self.index2split = [sum(self.outputs_dimension_per_outputs[:i])
                            for i in range(len(self.outputs_dimension_per_outputs) + 1)]
        self.acc = [tf.keras.metrics.Accuracy()
                    for i in range(len(self.outputs_dimension_per_outputs))]

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = self.split(y_true)
        y_pred = self.split(y_pred)

        for i in range(len(y_pred)):
            y_true_i = tf.argmax(y_true[i], axis=-1)
            y_pred_i = tf.argmax(y_pred[i], axis=-1)
            self.acc[i].update_state(y_true_i, y_pred_i)

    def result(self):
        return 0.0 #[acc.result() for acc in self.acc]

    def reset_states(self):
        for acc in self.acc:
            acc.reset_states()

    @tf.autograph.experimental.do_not_convert
    def split(self, inputs):
        return [inputs[:, self.index2split[i]: self.index2split[i + 1]] for i in range(len(self.index2split) - 1)]