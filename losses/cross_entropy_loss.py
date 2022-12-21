import tensorflow as tf


class CELoss(tf.keras.losses.Loss):

    def __init__(self, index_y_true, num_classes):
        super(CELoss, self).__init__()
        self.num_classes = num_classes
        self.ce = tf.keras.losses.CategoricalCrossentropy()
        self.index_y_true = index_y_true
        self.reshape = tf.keras.layers.Reshape(target_shape=(17,))
        self.indexes = [i for i in range(self.num_classes)]

    def call(self, y_true, y_pred):
        y_true = tf.squeeze(self.convert_matrix2one_hot(y_true), axis=1)
        return self.ce(y_true, y_pred)

    def convert_matrix2one_hot(self, y_true):
        y_true = tf.split(y_true, num_or_size_splits=16, axis=1)[self.index_y_true] # (b, 16, 17) -> (b, 1, 17)

        y_true = tf.squeeze(y_true, axis=1) # (b, 1, 17) -> (b, 17)
        y_true = tf.gather(y_true, indices=[self.indexes], axis=1)
        return tf.cast(y_true, dtype=tf.int32)

    def set_indexes(self):
        self.indexes = [i for i in range(self.num_classes)]