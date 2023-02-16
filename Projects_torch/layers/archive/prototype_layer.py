import tensorflow as tf


class Prototype(tf.keras.layers.Layer):

    def __init__(self,
                 num_labeled,
                 num_unlabeled,
                 num_channels,
                 threshold,
                 **kwargs):

        super().__init__(**kwargs)
        self.novelty = []
        self.labeled = None
        self.unlabeled = None
        self.num_labeled = num_labeled
        self.num_unlabeled = num_unlabeled
        self.num_channels = num_channels
        self.cos_sim = tf.keras.losses.CosineSimilarity(reduction=tf.keras.losses.Reduction.NONE)
        self.threshold = threshold

    def build(self, input_shape):

        self.labeled = [self.add_weight(shape=(self.num_channels,),
                                       trainable=True,
                                       initializer='random_normal',
                                       name='labeled')
                        for i in range(self.num_labeled)]

        self.unlabeled = [self.add_weight(shape=(self.num_channels,),
                                         trainable=True,
                                         initializer='random_normal',
                                         name='unlabeled')
                          for i in range(self.num_unlabeled)]

    @tf.function(input_signature=(tf.TensorSpec(shape=[None, ], dtype=tf.float32),))
    def unstack(self, inputs):
        return tf.unstack(inputs, axis=0)

    def call(self, inputs, **kwargs):
        inputs_list = tf.function(tf.unstack(inputs, axis=0))
        list_novelty = []
        list_labeled = []
        list_unlabeled = []

        for sample in inputs_list:

            arg_min_labeled_val = 1.5
            arg_min_labeled_index = 0

            arg_min_novelty_val = 1.5
            arg_min_novelty_index = 0

            arg_min_unlabeled_val = 1.5
            arg_min_unlabeled_index = 0

            for i in range(len(self.labeled)):
                cos_sim = tf.math.abs(self.cos_sim(self.labeled[i], sample))
                if cos_sim >= self.threshold:


    def calc_distance(self, sample, prototypes_list):
        arg_min_val = 1.5
        arg_min_index = 0

        for i in range(len(prototypes_list)):
            cos_sim = tf.math.abs(self.cos_sim(prototypes_list[i], sample))
            if cos_sim >= self.threshold:


