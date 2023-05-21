import os
import sys
import pickle
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


class DataLoaderSTFT(Dataset):
    def __init__(self, num_classes, masking=True, mask=None, norm_mean=None, norm_std=None, dataset_path=None):

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
        self.masking = masking

    def load_dataset(self, dataset_path):
        self.files = [os.path.join(dataset_path, file)
                      for file in os.listdir(dataset_path) if file.endswith('.wav')]
        if sys.platform == 'win32':
            self.labels = [file.replace('\data', '\labels').replace('.wav', '.npy')
                           for file in self.files]
        else:
            self.labels = [file.replace('data/', 'labels/').replace('.wav', '.npy')
                           for file in self.files]
        print(len(self.files), len(self.labels))
        self.files_ = list(zip(self.files[:256], self.labels[:256]))

    def __len__(self):
        return len(self.files_)

    def mask(self, x):
        mask_d = np.ones((65, 65), dtype=np.float32)
        for i in range(65):
            for j in range(65):
                if j > i:
                    mask_d[i][j] = 0
        return x, mask_d  # np.float32(mask_d)

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
        f, t, Zxx = signal.stft(data, samplerate, nperseg=253, nfft=256)  # nperseg=256)
        # data = (np.abs(Zxx) - self.norm_mean) / self.norm_std
        data = np.transpose(np.abs(Zxx))
        data = np.ndarray.astype(data, np.float32)

        label = self.label2onehot(label)
        label = np.ndarray.astype(label, np.float32)

        if self.masking:
            data = list(self.mask(data))

        return data, label