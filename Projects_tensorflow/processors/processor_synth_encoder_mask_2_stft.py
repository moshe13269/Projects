import os
import pickle
import dataset
import numpy as np
from typing import List, Tuple
import tensorflow as tf
from scipy.io import wavfile
from scipy import signal
import librosa


class Processor:
    std_mean_calc: dataset.StdMeanCalc
    """
    TalNoise:
    Mean: 0.026406644
    Std: 0.20221268
    
    Noy:
    Mean: 0.013329904 
    Std: 0.041720923
    """
    def __init__(self,
                 norm_mean,
                 norm_std,
                 num_classes,
                 std_mean_calc,
                 mask: Tuple[int, int, int]):
        self.num_classes = num_classes
        self.std_mean_calc = std_mean_calc
        # norm_vec_mean, norm_vec_std = self.std_mean_calc.load_dataset()
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.num_classes_per_outputs = np.asarray(
            [sum(num_classes[:i])
             for i in range(len(num_classes))]
        )
        self.mask_shape = mask

    def label2onehot(self, labels):
        onehot_labels = np.zeros(sum(self.num_classes))
        for i in range(labels.shape[0]):
            onehot_labels[int(labels[i]) + int(self.num_classes_per_outputs[i])] = 1.
        return onehot_labels

    @tf.function
    def mask(self, x):
        # mask_e = tf.where(tf.random.uniform(tuple(self.mask_shape)) >= 0.45, 1., 0.)
        # mask_d = tf.where(tf.random.uniform(tuple(self.mask_shape)) >= 0.45, 1., 0.)

        # tf.concat([tf.zeros((mask, mask)),
        #                    tf.ones((mask, (self.d_model // self.heads) - mask))
        #                    ],
        #                   axis=1)
        mask_e = np.zeros((65, 65)) #np.concatenate([np.zeros((65, 65)), np.ones((65, 63))])
        mask_d = np.ones((65, 65))
        for i in range(65):
            for j in range(65):
                if j > i:
                    mask_d[i][j] = 0

        return x, mask_d #x, mask_e, mask_d

    @tf.function
    def mask_inference(self, x):
        # mask_e = tf.ones(tuple(self.mask_shape))
        mask_d = tf.ones((65, 1))
        return x, mask_d #mask_e,

    def load_data(self, path2data):
        label = np.squeeze(np.load(path2data[1]))
        samplerate, data = wavfile.read(path2data[0])
        f, t, Zxx = signal.stft(data, samplerate, nperseg=253, nfft=256) #nperseg=256)
        data = (np.abs(Zxx) - self.norm_mean) / self.norm_std
        data = np.transpose(data)

        # data = np.expand_dims(Zxx, axis=-1)
        # Zxx = np.transpose(np.abs(Zxx))
        # data = librosa.feature.mfcc(y=data, sr=16384, n_mfcc=40) #40,33
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

