
import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras.losses import CosineSimilarity


class ContrastiveLoss(Loss):

    def __init__(self, temp=1.):
        super().__init__()
        self.temp = temp
        self.epsilon = 10 ** -10
        self.num_distractors = 4

    def call(self, y_true, y_pred):
        mask = y_true
        Q_t, c_t = tf.split(y_pred, 2, axis=-1)
        num_c_t = tf.cast(tf.reduce_max(tf.reduce_sum(mask, axis=-1), axis=-1), tf.int32)

        _, indexes_q_t_c_t = tf.math.top_k(mask, num_c_t)
        _, indexes_distractor = tf.math.top_k(mask, num_c_t + self.num_distractors)

        index_q_t = tf.expand_dims(tf.repeat(
            tf.expand_dims(tf.range(0, tf.shape(indexes_q_t_c_t)[0], delta=1), axis=1),
            tf.shape(indexes_q_t_c_t)[1], axis=1), axis=2)

        batch_index_q_t = tf.expand_dims(indexes_q_t_c_t, axis=2)
        indexes_q_t_c_t = tf.concat([index_q_t, batch_index_q_t], axis=2)

        index_distractor = tf.expand_dims(tf.repeat(
            tf.expand_dims(tf.range(0, tf.shape(indexes_distractor)[0], delta=1), axis=1),
            tf.shape(indexes_distractor)[1], axis=1), axis=2)

        batch_index_distractor = tf.expand_dims(indexes_distractor, axis=2)
        indexes_distractor = tf.concat([index_distractor, batch_index_distractor], axis=2)

        q_t = tf.gather_nd(Q_t, indexes_q_t_c_t)
        q_t_distractor = tf.gather_nd(Q_t, indexes_distractor)
        c_t = tf.gather_nd(c_t, indexes_q_t_c_t)

        numerator = (-CosineSimilarity(axis=-1, reduction='none')(c_t, q_t)) / self.temp
        clipped_numerator = tf.math.exp(tf.clip_by_value(numerator, clip_value_min=-1e100, clip_value_max=80))

        b, _, c = tf.shape(c_t)

        q_t_distractor = tf.tile(tf.expand_dims(q_t_distractor, 1), [1, num_c_t, 1, 1])
        q_t_distractor = tf.reshape(q_t_distractor, (b * num_c_t, num_c_t + self.num_distractors, c))
        c_t = tf.reshape(c_t, (num_c_t * b, 1, c))
        denominator = tf.reduce_sum(tf.math.exp(-CosineSimilarity(axis=-1, reduction='none')
        (q_t_distractor, c_t) / self.temp), axis=-1)

        return tf.reduce_mean(-tf.math.log(tf.reshape(clipped_numerator, (b * num_c_t,)) / denominator))


