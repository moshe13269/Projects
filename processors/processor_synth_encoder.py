import os
import pickle
import numpy as np
from typing import List
import tensorflow as tf
from scipy.io import wavfile


class Processor:
    # num_classes: List[int]

    def __init__(self, num_classes):
        self.num_classes = np.array(num_classes)
        # with open(path2sets, 'rb') as handle:
        #     self.list_of_sets = pickle.load(handle)

    def label2onehot(self, data, label):
        onehot_labels = []
        # for i in range(len(self.num_classes)):
        #     ohe = np.eye(self.num_classes[i])[
        #         int(np.asarray([label[i]]))]  # tf.keras.utils.to_categorical(label[i], num_classes=self.num_classes[i])
        #     onehot_labels.append(ohe.astype(np.float32))
        print(tf.shape(data), tf.shape(label))
        return data, label #tuple(onehot_labels)

    def load_data(self, path2data):
        label = np.squeeze(np.load(path2data[1]))
        samplerate, data = wavfile.read(path2data[0])
        data = data.reshape(data.shape[0], 1)
        # data = (data - np.mean(data)) / np.std(data)
        data = np.ndarray.astype(data, np.float32)
        # label = self.label2onehot(label)
        label = np.ndarray.astype(label/self.num_classes, np.float32)

        return data, label


if __name__ == '__main__':
    from dataset.dataset import Dataset

    dataset = Dataset(path2load='/home/moshelaufer/PycharmProjects/datasets/tal_noise_1',
                      labels=True, type_files='wav', dataset_names='tal_noise_1')
    num_classes: List[int] = [16, 16, 8, 8, 9, 8, 9, 9, 2, 4, 4, 4, 17, 17, 17, 17]
    processor = Processor(num_classes)
    train_dataset = tf.data.Dataset.from_tensor_slices((dataset.dataset_names_train[:10],
                                                        dataset.labels_names_train[:10]))

    train_dataset = (train_dataset
                     .shuffle(1024)
                     .map(lambda path2data, path2label:
                          tf.numpy_function(processor.load_data, [(path2data, path2label)],
                                            [tf.float32, tf.float32])
                          , num_parallel_calls=tf.data.AUTOTUNE)
                     .cache()
                     .batch(2)
                     .prefetch(tf.data.AUTOTUNE)
                     )

    # train_dataset = train_dataset.map(lambda x, y:
    #                                   tf.numpy_function(processor.label2onehot, [(x, y)],
    #                                                     [tf.float32, tf.float32]))

    iterator = train_dataset.as_numpy_iterator()
    for i in range(1):
        d = iterator.next()
        print(d[1])
        # print(d.shape)

# [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32
#                                              , tf.float32, tf.float32, tf.float32, tf.float32
#                                              , tf.float32, tf.float32, tf.float32, tf.float32
# , tf.float32, tf.float32, tf.float32, tf.float32]
