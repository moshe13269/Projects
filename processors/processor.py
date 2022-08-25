import numpy as np
import tensorflow as tf
from tensorflow import keras


class Processor(keras.utils.Sequence):

    def __init__(self, x_in=None, batch_size=512, num2mask=10, t_axis=138, y_in=None, shuffle=True):
        # Initialization
        self.num2mask = num2mask
        self.t_axis = t_axis
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.x = np.arange(1, 1000) #x_in
        self.y = y_in
        self.datalen = len(self.x)
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

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)