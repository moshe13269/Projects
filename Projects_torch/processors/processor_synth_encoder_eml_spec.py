import os
import sys
import pickle
import librosa
import dataset
import torchvision
import numpy as np
from scipy import signal
from random import shuffle
from scipy.io import wavfile
from typing import List, Tuple
from torch.utils.data import Dataset

"""
    TalNoise:
    Mean: 0.026406644
    Std: 0.20221268

    Noy:
    Mean: 0.013329904 
    Std: 0.041720923
    """


class DataLoaderMelSpec(Dataset):
    def __init__(self, num_classes, autoencoder=False, norm_mean=None, norm_std=None, dataset_path=None,
                 calc_mean=False):

        super().__init__()
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.dataset_path = None
        self.num_classes = num_classes
        self.num_classes_per_outputs = np.asarray(
            [sum(num_classes[:i])
             for i in range(len(num_classes))]
        )
        self.files = None
        self.files_ = None
        self.labels = None
        self.calc_mean = calc_mean
        self.autoencoder = autoencoder

    def load_dataset(self, dataset_path):
        self.files = [os.path.join(dataset_path, file)
                      for file in os.listdir(dataset_path) if file.endswith('.wav')]

        if sys.platform == 'win32':
            self.labels = [file.replace('\data', '\labels').replace('.wav', '.npy')
                           for file in self.files]
        else:
            self.labels = [file.replace('data/', 'labels/').replace('.wav', '.npy')
                           for file in self.files]
        self.files_ = list(zip(self.files, self.labels))

    def __len__(self):
        return len(self.files_)

    def shuffle_(self):
        shuffle(self.files)

    def label2onehot(self, labels):
        onehot_labels = np.zeros(sum(self.num_classes))
        for i in range(labels.shape[0]):
            onehot_labels[int(labels[i]) + int(self.num_classes_per_outputs[i])] = 1.
        return onehot_labels

    def __getitem__(self, idx):
        wav_file = self.files_[idx][0]
        if sys.platform == 'win32':
            label_file = self.files_[idx][0].replace('\data', '\labels').replace('.wav', '.npy')
        else:
            label_file = self.files_[idx][0].replace('data/', 'labels/').replace('.wav', '.npy')
        label = np.squeeze(np.load(label_file))
        samplerate, data = wavfile.read(wav_file)

        sgram = librosa.stft(data)
        sgram_mag, _ = librosa.magphase(sgram)
        mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=samplerate)
        mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)

        if self.calc_mean:
            return mel_sgram.mean(), mel_sgram.std()
        # data = np.transpose(data)
        # data = np.ndarray.astype(data, np.float32)

        # label = self.label2onehot(label)
        label = np.split(label, label.shape[0])
        label = [np.ndarray.astype(label_, dtype=np.int64)
                  for label_ in label]
        # label = np.ndarray.astype(np.expand_dims(label_, 1), np.float32)

        if self.autoencoder:
            return mel_sgram, [mel_sgram, label]
        return mel_sgram, label


if __name__ == '__main__':
    num_classes = [3, 12, 20, 31, 4, 5, 8, 5, 16]
    dl = DataLoaderMelSpec(num_classes=num_classes, autoencoder=False, calc_mean=True)
    dl.load_dataset(dataset_path=r'C:\Users\moshe\PycharmProjects\commercial_synth_dataset\noy\data')
    mean = 0.0
    std = 0.0
    for i in range(len(dl)):
        mean_, std_ = dl.__getitem__(i)
        mean += mean_
        std += std_
        if i % 1000 == 0:
            print(i)
    print('Mean: %f10, Std: %f10' % (mean, std))



    # def mask(self, x):
    #     mask_d = np.ones((65, 65), dtype=np.float32)
    #     for i in range(65):
    #         for j in range(65):
    #             if j > i:
    #                 mask_d[i][j] = 0
    #     return x, mask_d  # np.float32(mask_d)