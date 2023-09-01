import os
import sys
import pickle
import dataset
import librosa
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
    def __init__(self, num_classes, norm_mean=None, norm_std=None, dataset_path=None):

        super().__init__()
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.dataset_path = None
        self.num_classes_per_outputs = np.asarray(
            [sum(num_classes[:i])
             for i in range(len(num_classes))]
        )
        self.files = None
        self.files_ = None
        self.labels = None

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

        samples, sample_rate = librosa.load(wav_file, sr=None)
        sgram = librosa.stft(samples, win_length=64)
        sgram_mag, _ = librosa.magphase(sgram)
        mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)
        mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)

        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(14, 14))
        # librosa.display.specshow(mel_sgram, sr=sample_rate, x_axis='time', y_axis='mel')
        # plt.colorbar(format='%+2.0f dB')


        # data = (np.abs(Zxx) - 0.013329904)/0.041720923
        # data = np.transpose(data)

        data = np.ndarray.astype(mel_sgram, np.float32)
        label = np.ndarray.astype(label, np.int_)

        return [data, label], data


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    sample_rate = 16384
    num_classes = [3, 12, 20, 31, 4, 5, 8, 5, 16]
    dl = DataLoaderSTFT(num_classes=num_classes)
    dl.load_dataset(dataset_path=r'C:\Users\moshe\PycharmProjects\commercial_synth_dataset\noy\data')
    a = dl.__getitem__(754)
    b = 0



    # def mask(self, x):
    #     mask_d = np.ones((65, 65), dtype=np.float32)
    #     for i in range(65):
    #         for j in range(65):
    #             if j > i:
    #                 mask_d[i][j] = 0
    #     return x, mask_d  # np.float32(mask_d)