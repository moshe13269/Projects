import os
import pickle
import dataset
import numpy as np
from typing import List
import tensorflow as tf
from scipy.io import wavfile


class Processor:
    std_mean_calc: dataset.StdMeanCalc

    def __init__(self, num_classes, std_mean_calc):
        self.num_classes = len(num_classes)
        self.std_mean_calc = std_mean_calc
        norm_vec_mean, norm_vec_std = self.std_mean_calc.load_dataset()
        print(norm_vec_mean, norm_vec_std)
        self.norm_vec_mean = norm_vec_mean
        self.norm_vec_std = norm_vec_std
        # with open(path2sets, 'rb') as handle:
        #     self.list_of_sets = pickle.load(handle)

    def label2onehot(self, labels):
        onehot_labels = np.zeros((self.num_classes, 17))
        for i in range(labels.shape[0]):
            onehot_labels[i][int(labels[i])] = 1
        return onehot_labels
        # onehot_labels = []
        # for label in range(labels):
        #     ohe = np.eye(self.num_classes)[int(np.asarray([label]))]
        #     onehot_labels.append(ohe.astype(np.float32))
        # # print(tf.shape(data), tf.shape(label))
        # return np.asarray(onehot_labels)

    def load_data(self, path2data):
        label = np.squeeze(np.load(path2data[1]))
        samplerate, data = wavfile.read(path2data[0])
        # data = np.log(data + 10**-10)
        data = (data - self.norm_vec_mean) / (self.norm_vec_std + 10**-20)
        data = data.reshape(data.shape[0], 1)
        data = np.ndarray.astype(data, np.float32)
        label = self.label2onehot(label)
        label = np.ndarray.astype(label, np.float32)

        return data, label


if __name__ == '__main__':
    from dataset.dataset import Dataset
    from dataset.calc_std_mean_dataset_wav import StdMeanCalc

    dataset = Dataset(path2load=['/home/moshelaufer/PycharmProjects/datasets/tal_noise_1'],
                      labels=True, type_files='wav', dataset_names='tal_noise_1')

    num_classes: List[int] = [16, 16, 8, 8, 9, 8, 9, 9, 2, 4, 4, 4, 17, 17, 17, 17]

    std_mean_calc = StdMeanCalc('/home/moshelaufer/PycharmProjects/datasets/tal_noise_25000_base/')

    processor = Processor(num_classes, std_mean_calc)
    train_dataset = tf.data.Dataset.from_tensor_slices((dataset.dataset_names_train[:10],
                                                        dataset.labels_names_train[:10]))

    train_dataset = (train_dataset
                     .shuffle(1024)
                     .map(lambda path2data, path2label:
                          tf.numpy_function(processor.load_data, [(path2data, path2label)],
                                            [tf.float32, tf.float32])
                          , num_parallel_calls=tf.data.AUTOTUNE)#.map(lambda x, y: (x, tuple([y for i in range(4)])))
                     .cache()
                     .batch(1)
                     .prefetch(tf.data.AUTOTUNE)
                     )

    # train_dataset = train_dataset.map(lambda x, y:
    #                                   tf.numpy_function(processor.label2onehot, [(x, y)],
    #                                                     [tf.float32, tf.float32]))

    iterator = train_dataset.as_numpy_iterator()
    for i in range(5):
        d = iterator.next()
        print(d[1])
        # print(tf.reduce_sum(tf.where(d[1][0] == d[1][1], 0., 1.)))
        # print(tf.reduce_sum(tf.where(d[1][2] == d[1][3], 0., 1.)))
        # # print(d[1])

