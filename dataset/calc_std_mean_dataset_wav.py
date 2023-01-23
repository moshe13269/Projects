import numpy as np
from os import listdir
from os.path import isfile, join
from scipy.io import wavfile
from scipy import signal


class StdMeanCalc:
    path2dataset: str

    def __init__(self, path2dataset, stft=False):
        self.path2dataset = path2dataset
        self.list_of_dataset_files = [join(path2dataset, f) for f in listdir(path2dataset)
                                      if isfile(join(path2dataset, f))
                                      and join(path2dataset, f).endswith('.wav')]
        self.std = None
        self.mean = None
        self.dataset = []
        self.stft = stft

    def load_dataset(self):
        if not self.stft:
            for file in self.list_of_dataset_files:
                _, data = wavfile.read(file)
                self.dataset.append(data)
        else:
            for file in self.list_of_dataset_files:
                samplerate, data = wavfile.read(file)
                f, t, Zxx = signal.stft(data, samplerate, nperseg=256, nfft=512)
                data = np.abs(Zxx)
                self.dataset.append(data)

        self.dataset = np.asarray(self.dataset)
        # self.std = self.dataset.std()
        # self.mean = self.dataset.mean()

        np.save(arr=self.std, file=join(self.path2dataset, 'std'))
        np.save(arr=self.mean, file=join(self.path2dataset, 'mean'))
        std_mean = self.std.mean()
        mean_mean = self.mean.mean()
        return mean_mean, std_mean
        # print('Mean: %10.f, Std: %10.f' % (mean_mean, std_mean))
        # print('Std: %10.f' % (self.dataset.std()))


if __name__ == '__main__':
    calc = StdMeanCalc('/home/moshelaufer/PycharmProjects/datasets/tal_noise_25000_base/')
    calc.load_dataset()

