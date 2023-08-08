import os
import sys
import torch
import numpy as np
from random import shuffle
from scipy.io import wavfile
from torch.utils.data import Dataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class DataLoader(Dataset):
    def __init__(self, dataset_path=None
                 ):

        super().__init__()
        self.dataset_path = dataset_path

        if self.dataset_path is not None:
            self.load_dataset()
        else:
            self.files = None
            self.files_ = None
            self.labels = None

    def load_dataset(self, path=None):
        if path is None:
            dataset_path = self.dataset_path
        else:
            dataset_path = path
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

    def __getitem__(self, idx):
        wav_file = self.files_[idx][0]
        if sys.platform == 'win32':
            label_file = self.files_[idx][0].replace('\data', '\labels').replace('.wav', '.npy')
        else:
            label_file = self.files_[idx][0].replace('data/', 'labels/').replace('.wav', '.npy')
        label = np.squeeze(np.load(label_file))
        _, data = wavfile.read(wav_file)

        # data = torch.unsqueeze(torch.from_numpy(data), dim=0)

        return torch.from_numpy(data), torch.from_numpy(np.int64(label))
