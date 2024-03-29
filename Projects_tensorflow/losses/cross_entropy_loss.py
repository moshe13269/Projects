import tensorflow as tf


class CELoss(tf.keras.losses.Loss):

    def __init__(self, index_y_true, num_classes):
        super(CELoss, self).__init__()
        self.num_classes = num_classes
        self.ce = tf.keras.losses.CategoricalCrossentropy()
        self.index_y_true = index_y_true
        self.reshape = tf.keras.layers.Reshape(target_shape=(17,))
        self.indexes = [i for i in range(self.num_classes)]
        # self.outputs_dimension_per_outputs = outputs_dimension_per_outputs

    def call(self, y_true, y_pred):
        # tf.print(tf.shape(y_pred), tf.shape(y_true))
        # y_true = tf.squeeze(self.convert_matrix2one_hot(y_true), axis=1)
        y_true = self.convert_matrix2one_hot(y_true)
        # tf.print(tf.shape(y_pred), tf.shape(y_true))
        # return tf.reduce_mean(y_pred)
        # y_pred = tf.nn.softmax(y_pred)
        # return tf.reduce_mean(-tf.math.log(tf.reduce_sum(y_true * y_pred, axis=-1)))

        # return tf.reduce_mean(tf.reduce_sum(- tf.cast(y_true, dtype=tf.float32) * y_pred, axis=-1))
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)) #self.ce(y_true, y_pred)

    @tf.autograph.experimental.do_not_convert
    def convert_matrix2one_hot(self, y_true):
        y_true = tf.split(y_true, num_or_size_splits=9, axis=1)[self.index_y_true]  # (b, 16, 17) -> (b, 1, 17)

        y_true = tf.squeeze(y_true, axis=1)  # (b, 1, 17) -> (b, 17)
        y_true = tf.gather(y_true, indices=self.indexes, axis=1)
        return tf.cast(y_true, dtype=tf.int32)

    def set_indexes(self):
        self.indexes = [i for i in range(self.num_classes)]
