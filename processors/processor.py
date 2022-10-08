import pickle
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
    def create_mask(self):
        rand_uniform = tf.random.uniform(maxval=1, shape=(self.t_axis,))
        mask = tf.where(
            tf.sign(rand_uniform - tf.reduce_min(tf.math.top_k(rand_uniform, k=self.point2mask)[0])) >= 0,
            1., 0.)
        return mask

    def load_data(self, path2data):
        if self.load_label:
            path2label = path2data.replace('data', 'labels')
            label_file = open(path2label, 'r')
            label = pickle.load(label_file)
            # todo: convert to onehot vector
        else:
            label = self.create_mask()

        wav_file = tf.io.read_file(path2data[0])
        wav_file = (wav_file - tf.reduce_mean(wav_file)) / tf.math.reduce_std(wav_file)
        wav_file = tf.audio.decode_wav(wav_file)

        return [wav_file, label], [label]
