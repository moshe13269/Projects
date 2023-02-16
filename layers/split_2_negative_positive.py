import tensorflow as tf


class SplitNegativePositive(tf.keras.layers.Layer):

    def __init__(self,
                 # prototype_l,
                 # prototype_u,
                 threshold,
                 **kwargs):

        super().__init__(**kwargs)
        self.threshold = threshold
        self.cos_sim = tf.keras.losses.CosineSimilarity(reduction=tf.keras.losses.Reduction.NONE)

    def call(self, inputs, **kwargs):
        # inputs: (Batch, channels), prototype_l: (K, channels), prototype_u: (M, channels)
        prototype_l, prototype_u, inputs = inputs

        prototype_l_candidates = []
        prototype_u_candidates = []

        # prototype_u_list = tf.split(prototype_u, axis=0, num_or_size_splits=tf.shape(prototype_u)[0])
        # prototype_l_list = tf.split(prototype_l, axis=0, num_or_size_splits=tf.shape(prototype_l)[0])

        prototype_u_list = tf.unstack(prototype_u, axis=0)
        prototype_l_list = tf.unstack(prototype_l, axis=0)
        inputs_list = tf.unstack(inputs, axis=0)

        for input_vector in inputs_list:
            flag = 0
            for u_vector in prototype_l_list:
                if self.cos_sim(u_vector, input_vector) >= self.threshold:
                    prototype_l_candidates.append(input_vector)
                    flag = 1
                if flag == 1:
                    break
            if flag != 1:
                for l_vector in prototype_u_list:
                    if self.cos_sim(l_vector, input_vector) >= self.threshold:
                        prototype_u_candidates.append(input_vector)

        return tf.concat(prototype_u_candidates), tf.concat(prototype_l_candidates)
