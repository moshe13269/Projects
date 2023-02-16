import tensorflow as tf


class MultiCELoss(tf.keras.losses.Loss):

    def __init__(self, num_params, num_classes_per_param):
        super(MultiCELoss, self).__init__()
        self.ce = tf.keras.losses.CategoricalCrossentropy()
        self.num_params = num_params
        self.num_classes_per_param = num_classes_per_param

    def call(self, y_true, y_pred):

        y_trues = tf.split(y_true, num_or_size_splits=self.num_params, axis=1)
        b = tf.shape(y_true)
        # tf.print(b)
        loss = 0.0
        for i in range(16):
            y_pred_i = tf.gather_nd(tf.concat(tf.range(0, b, 1), y_trues[i]), params=y_pred[i])
            loss += tf.reduce_mean(-tf.math.log(y_pred_i))
        return tf.reduce_mean(y_pred)



        # y_true = self.convert_class_index_2_one_hot(y_true)
        #
        # loss = []
        #
        # # return tf.reduce_mean(y_pred)
        # for i in range(self.num_params):
        #     loss.append(self.ce(y_true[i], y_pred[i]))
        # return tf.reduce_mean(loss)

    def convert_class_index_2_one_hot(self, y_true):
        y_trues = tf.split(y_true, num_or_size_splits=self.num_params, axis=1)
        # tf.print(tf.shape(y_true), tf.shape(y_trues[0]), len(y_trues))
        # tf.print(self.num_classes_per_param)
        a = []
        for i in range(self.num_params):
            a.append(tf.one_hot(tf.cast(tf.squeeze(y_trues[i]), tf.int32), self.num_classes_per_param[i])) #tf.keras.utils.to_categorical(tf.cast(y_trues[i], tf.int32), num_classes=self.num_classes_per_param[i]))
        return a
