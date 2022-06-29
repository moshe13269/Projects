import numpy as np
import tensorflow as tf
from tensorflow import keras


class Processor(keras.utils.Sequence):

    def __init__(self, x_in, batch_size, num2mask, t_axis, y_in=None, shuffle=True):
        # Initialization
        self.num2mask = num2mask
        self.t_axis = t_axis
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.x = x_in
        self.y = y_in
        self.datalen = len(x_in)
        self.indexes = np.arange(self.datalen)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def craete_mask(self, batch_shape):
        random = tf.random.normal(shape=(batch_shape, self.t_axis))
        point2mask, _ = tf.math.top_k(random, self.num2mask)
        mask = tf.where(random - tf.reduce_min(point2mask, axis=-1, keepdims=True) >= 0., 1., 0.)
        return mask

    def __getitem__(self, index):
        # get batch indexes from shuffled indexes
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        x_batch = self.x[batch_indexes]
        y_batch = self.craete_mask(x_batch[0].shape[0])
        return x_batch, y_batch

    def __len__(self):
        # Denotes the number of batches per epoch
        return self.datalen // self.batch_size

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(self.datalen)
        if self.shuffle:
            np.random.shuffle(self.indexes)