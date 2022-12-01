
import os
import numpy as np
import tensorflow as tf
from scipy.io import wavfile


class Processor:

    def __init__(self, path2num_classes):
        self.num_classes = np.load(path2num_classes)

    def label2onehot(self, label):
        onehot_labels = []
        for i in range(label.shape[0]):
            onehot_labels.append(tf.keras.utils.to_categorical(label[i], num_classes=self.num_classes[i]))
        return np.asarray(onehot_labels)

    def load_data(self, path2data):
        label = np.load(path2data[1])
        samplerate, data = wavfile.read(path2data[0])
        data = data.reshape(data.shape[0], 1)
        data = (data - np.mean(data)) / np.std(data)
        data = np.ndarray.astype(data, np.float32)
        label = self.label2onehot(label)
        label = np.ndarray.astype(label, np.float32)

        return data, label


if __name__ == '__main__':
    from dataset.dataset import Dataset

    dataset = Dataset(path2load=r'C:\Users\moshe\PycharmProjects\datasets\dx_wav',
                      labels=False, type_files='wav', dataset_names='dx_wav')
    processor = Processor(t_axis=50, prob2mask=0.065, masking_size=10, load_label=False)
    train_dataset = tf.data.Dataset.from_tensor_slices(dataset.dataset_names_train[:10])

    train_dataset = (train_dataset
                     .shuffle(1024)
                     .map(lambda item: tf.numpy_function(processor.load_data, [item], [tf.float32, tf.float32])
                          , num_parallel_calls=tf.data.AUTOTUNE).
                     map(lambda x, y: ((x, y), y)).cache().batch(2).prefetch(tf.data.AUTOTUNE))

    iterator = train_dataset.as_numpy_iterator()
    for i in range(1):
        d = iterator.next()
        print(d)
        print(d.shape)
