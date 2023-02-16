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

        self.classes_weight = {'8': [0.750588, 0.249412],
                               '10': [0.211888, 0.363552, 0.363552, 0.363552],
                               '11': [0.0375, 0.0375, 0.0375, 0.886572],
                               '12': [0.150476, 0.150476, 0.150476, 0.150476, 0.150476, 0.150476, 0.0537, 0.150476,
                                      0.150476, 0.150476, 0.150476, 0.150476, 0.150476, 0.150476, 0.150476, 0.150476,
                                      0.150476],
                               '13': [0.009392, 0.849824, 0.849824, 0.849824, 0.849824, 0.849824, 0.849824, 0.849824,
                                      0.849824,
                                      0.849824, 0.849824, 0.849824, 0.849824, 0.849824, 0.849824, 0.849824, 0.849824],
                               '14': [0.150476, 0.150476, 0.150476, 0.150476, 0.150476, 0.053316, 0.150476, 0.150476,
                                      0.150476,
                                      0.150476, 0.150476, 0.150476, 0.150476, 0.150476, 0.150476, 0.150476, 0.150476],
                               '15': [0.00922, 0.849824, 0.849824, 0.849824, 0.849824, 0.849824, 0.849824, 0.849824,
                                      0.849824,
                                      0.849824, 0.849824, 0.849824, 0.849824, 0.849824, 0.849824, 0.849824, 0.849824]
                               }
        self.classes_with_weights = [8, 10, 11, 12, 13, 14, 15]

    def call(self, y_true, y_pred):
        # tf.print(tf.shape(y_true), tf.shape(y_true))
        y_true = self.split(y_true)
        y_pred = self.split(y_pred)

        # y_pred = [tf.nn.softmax(output) for output in y_pred]

        # loss = [tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true[i], logits=y_pred[i]))
        #         for i in range(len(y_pred))]
        loss = 0.0
        for i in range(len(y_pred)):
            # if i in self.classes_with_weights and len(y_pred) > 9:
            #     loss += self.ce(y_true[i], tf.nn.softmax(y_pred[i]),
            #                     sample_weight=tf.constant(self.classes_weight[str(i)]))
            # else:
                loss += self.ce(y_true[i], tf.nn.softmax(y_pred[i]))
            # loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true[i], logits=y_pred[i]))
        loss = loss / len(y_pred)
        return loss

    @tf.autograph.experimental.do_not_convert
    def split(self, inputs):
        return [inputs[:, self.index2split[i]: self.index2split[i + 1]] for i in range(len(self.index2split) - 1)]
        # tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1))


# self.classes_weight = {'8': [0.249412, 0.750588],
#                                '10': [0.363552, 0.21244, 0.21212, 0.211888],
#                                '11': [0.886572, 0.038316, 0.0375, 0.037612],
#                                '12': [0.05282, 0.053636, 0.053152, 0.052848, 0.05366, 0.053264, 0.150476, 0.052928,
#                                       0.052396, 0.05334, 0.053264, 0.05254, 0.052172, 0.052232, 0.054148, 0.053424,
#                                       0.0537],
#                                '13': [0.849824, 0.009332, 0.009344, 0.009084, 0.009492, 0.009456, 0.00946, 0.009568,
#                                       0.009392,
#                                       0.009324, 0.00932, 0.00928, 0.009812, 0.009508, 0.009104, 0.009308, 0.009392],
#                                '14': [0.05328, 0.052432, 0.053056, 0.05332, 0.0521, 0.150476, 0.05392, 0.053296,
#                                       0.053624,
#                                       0.054108, 0.053188, 0.053396, 0.051788, 0.052444, 0.053888, 0.052368, 0.053316],
#                                '15': [0.849824, 0.009128, 0.009296, 0.009228, 0.00922, 0.009296, 0.009304, 0.009568,
#                                       0.009568,
#                                       0.009416, 0.009648, 0.009412, 0.009288, 0.009348, 0.009556, 0.009424, 0.009476]
#                                }