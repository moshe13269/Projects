import tensorflow as tf


class CELoss(tf.keras.losses.Loss):
    """
    inputs: (batch, c') where c'=sum(outputs_dimension_per_outputs)

    """
    def __init__(self, outputs_dimension_per_outputs,
                 num_classes,
                 index_y_true):
        super(CELoss, self).__init__()
        self.num_classes = num_classes
        self.index_y_true = index_y_true

        self.outputs_dimension_per_outputs = outputs_dimension_per_outputs
        self.index2split = [sum(self.outputs_dimension_per_outputs[:i])
                            for i in range(len(self.outputs_dimension_per_outputs) + 1)]

        self.ce = tf.keras.losses.CategoricalCrossentropy()

    def call(self, y_true, y_pred):
        # tf.print(tf.shape(y_true), tf.shape(y_true))
        y_true = self.split(y_true)
        y_pred = self.split(y_pred)

        y_pred = [tf.nn.softmax(output) for output in y_pred]

        loss = [self.ce(y_true[i], y_pred[i]) for i in range(len(y_pred))]
        return sum(loss)

    @tf.function
    def split(self, inputs):
        return [inputs[:, self.index2split[i]: self.index2split[i + 1]] for i in range(len(self.index2split)-1)]
        # tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1))