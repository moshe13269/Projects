import os
import sys
import torch
# import pickle
import librosa
# import dataset
# import torchvision
import numpy as np
# from scipy import signal
from random import shuffle
from scipy.io import wavfile
import torch.nn.functional as F
# from typing import List, Tuple
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
    TalNoise:
    Mean: 0.026406644
    Std: 0.20221268

    Noy:
    Mean: 0.013329904 
    Std: 0.041720923
    """


class DataLoaderMelSpec(Dataset):
    def __init__(self,
                 num_classes,
                 win_length=256,
                 n_fft=1025,
                 encoder=False,
                 autoencoder=True,
                 norm_mean=None,
                 norm_std=None,
                 dataset_path=None,
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
        self.encoder = encoder
        self.win_length=win_length
        self.n_fft=n_fft

    def load_dataset(self, dataset_path):
        self.files = [os.path.join(dataset_path, file)
                      for file in os.listdir(dataset_path) if file.endswith('.wav')]

        if sys.platform == 'win32':
            self.labels = [file.replace(r'\\data\\', r'\\labels\\').replace('.wav', '.npy')
                           for file in self.files]
        else:
            self.labels = [file.replace('//data//', '//labels//').replace('.wav', '.npy')
                           for file in self.files]
        self.files_ = list(zip(self.files, self.labels))

    def __len__(self):
        return len(self.files_)

    def shuffle_(self):
        shuffle(self.files)

    # def label2onehot(self, labels):
    #     onehot_labels = np.zeros(sum(self.num_classes))
    #     for i in range(labels.shape[0]):
    #         onehot_labels[int(labels[i]) + int(self.num_classes_per_outputs[i])] = 1.
    #     return onehot_labels

    def label2onehot(self, label):
        onehot_label = []
        for i in range(label.shape[0]):
            onehot_label.append(
                F.one_hot(torch.tensor(label[i], dtype=torch.int64),
                          num_classes=max(self.num_classes))
            )
        onehot_label = torch.concatenate([torch.unsqueeze(tensor, dim=0) for tensor in onehot_label], dim=0)
        onehot_label = onehot_label.type(torch.float32)
        return onehot_label + torch.normal(torch.zeros_like(onehot_label), std=0.01)

    def __getitem__(self, idx):
        wav_file = self.files_[idx][0]
        if sys.platform == 'win32':
            label_file = self.files_[idx][0].replace(r'\\data\\', r'\\labels\\').replace('.wav', '.npy')
        else:
            label_file = self.files_[idx][0].replace('fm/data', 'fm/labels').replace('.wav', '.npy')
            # label_file = self.files_[idx][0].replace('//data//', '//labels//').replace('.wav', '.npy')
        label = np.squeeze(np.load(label_file))
        # label = self.label2onehot(label)
        samplerate, data = wavfile.read(wav_file)

        sgram = librosa.stft(data, win_length=self.win_length, n_fft=self.n_fft)
        sgram_mag, _ = librosa.magphase(sgram)
        mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=samplerate)
        mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)

        # if self.calc_mean:
        #     return mel_sgram.mean(), mel_sgram.std()
        # data = np.transpose(data)
        # data = np.ndarray.astype(data, np.float32)

        # label = self.label2onehot(label)

        # mel_sgram = (mel_sgram - 64.08) / 19.48
        # label = np.split(label, label.shape[0])
        # label = [np.ndarray.astype(label_, dtype=np.int64)
        #          for label_ in label]
        # label = np.ndarray.astype(np.expand_dims(label_, 1), np.float32)

        mel_sgram = (mel_sgram - mel_sgram.mean()) / mel_sgram.std()

        if self.encoder:
            # label_list = [torch.nn.functional.one_hot(torch.from_numpy(np.asarray(np.int64(label[j]))),
            #                                           self.num_classes[j]).type(torch.float32)
            #               for j in range(len(self.num_classes))]
            # mel_sgram = torch.cat((torch.from_numpy(mel_sgram), torch.mean(torch.from_numpy(mel_sgram), 1, True)), 1)
            return mel_sgram, torch.from_numpy(np.int64(label)) # (128, 256)

        label = torch.nn.functional.one_hot(torch.from_numpy(np.int64(label)), max(self.num_classes))
        label = label.type(torch.float32)
        if self.autoencoder:
            return [mel_sgram, label], mel_sgram
        return mel_sgram, label


if __name__ == '__main__':
    num_classes = [3, 12, 20, 31, 4, 5, 8, 5, 16]
    dl = DataLoaderMelSpec(num_classes=num_classes, autoencoder=False, calc_mean=False)
    dl.load_dataset(dataset_path=r'C:\Users\moshe\PycharmProjects\commercial_synth_dataset\noy\data')
    mean = 0.0
    std = 0.0
    for i in range(len(dl)):
        stft, label = dl.__getitem__(i)
        librosa.display.specshow(stft, sr=16384, x_axis='time', y_axis='mel')
        a = 0
        # mean_, std_ = dl.__getitem__(i)
        # mean += mean_
        # std += std_
        # if i % 1000 == 0:
        #     print(i)
        #     print('Mean: %f10, Std: %f10' % (mean / i, std / i))
    print('\n')
    print('--------------------------------------------------------')
    print('Mean: %f10, Std: %f10' % (mean / len(dl), std / len(dl)))
    print('--------------------------------------------------------')

    # import torch
    # import torchaudio
    # import torchaudio.transforms as T
    # import matplotlib.pyplot as plt
    # import os
    #
    # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    #
    # p = r'C:\Users\moshe\PycharmProjects\commercial_synth_dataset\dataset\2\wav'
    # w = os.path.join(p, '6.wav')
    # wav, f = torchaudio.load(w)
    # transform = T.AmplitudeToDB(stype="amplitude", top_db=80)
    #
    # waveform_db = transform(torch.stft(wav, 1024))

    # mel_spectrogram = T.MelSpectrogram(
    #     sample_rate=f,
    #     n_fft=1024,
    #     win_length=None,
    #     hop_length=512,
    #     center=True,
    #     pad_mode="reflect",
    #     power=2.0,
    #     norm='slaney',
    #     onesided=True,
    #     n_mels=128,
    #     mel_scale="htk",
    # )
    # melspec = mel_spectrogram(wav)
    # plt.plot(melspec[0])
    # plt.show()
    # a=0

    # def mask(self, x):
    #     mask_d = np.ones((65, 65), dtype=np.float32)
    #     for i in range(65):
    #         for j in range(65):
    #             if j > i:
    #                 mask_d[i][j] = 0
    #     return x, mask_d  # np.float32(mask_d)
