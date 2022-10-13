
import pickle
import numpy as np
import tensorflow as tf


class Processor:
    t_axis: int
    prob2mask: float
    masking_size: int
    top_k_transformer: int
    load_label: bool

    def __init__(self,
                 t_axis: int,
                 prob2mask: float,
                 masking_size: int,
                 load_label: bool,
                 # top_k_transformer: int,
                 ):

        self.t_axis = t_axis
        self.prob2mask = prob2mask
        self.point2mask = int(self.prob2mask * self.t_axis)
        self.masking_size = masking_size
        self.load_label = load_label
        # self.top_k_transformer = top_k_transformer

    # the masking area is '1' and the unmasking by '0'
    @tf.function(autograph=True)
    def create_mask(self):
        rand_uniform = tf.random.uniform(maxval=1, shape=(self.t_axis,))
        mask = tf.where(
            tf.sign(rand_uniform - tf.reduce_min(tf.math.top_k(rand_uniform, k=self.point2mask)[0])) >= 0.,
            1., 0.)
        return mask

    @tf.function(autograph=True)
    def load_data(self, path2data):
        if self.load_label:
            path2label = path2data.replace('data', 'labels')
            label_file = open(path2label, 'r')
            label = pickle.load(label_file)
            # todo: convert to onehot vector
        else:
            label = self.create_mask()

        # image = np.load(path2data)
        image = tf.io.read_file(path2data)
        image = tf.image.decode_image(image, channels=3, dtype=tf.float32)
        # image = tf.cast(tf.image.decode_image(image, channels=3, dtype=tf.float32), dtype=tf.float32)
        # tf.print(path2data)

        if tf.reduce_max(image) > 1.:
            image = image / 255.

        return (image, label), (label)
