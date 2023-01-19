import os
import sys
from os import listdir
from os.path import isfile, join
from dataclasses import dataclass
import pandas as pd
import numpy as np

"""
1. dataset folder contained a two folders: dataset and labels
2. dataset are samples of dataset and labels are samples of labels
3. the names of the samples in the both folders are same
"""


@dataclass
class Dataset:
    path2data: str
    path2csv: str
    path2save_labels: str

    """class for load path to samples of dataset and the labels (if had a labels)"""

    def __init__(self,
                 path2data,
                 path2csv,
                 path2save_labels
                 ):

        self.csv_file = pd.read_csv(path2csv)
        self.path2data = path2data
        self.path2save_labels = path2save_labels
        path2data_list = listdir(path2data)
        self.path2data_list = [file for file in path2data_list if file.endswith('.wav')]

        path2data_list_int = [int(f.replace('.wav', '')) for f in self.path2data_list]
        indexes_labels = list(self.csv_file['wav_id'])
        intersection = list(set(path2data_list_int) & set(indexes_labels))
        intersection_str = [str(index)+'.wav' for index in intersection]
        self.path2data_list = intersection_str

        assert self.csv_file.shape[0] == len(self.path2data_list), \
            print('Num of label: %d and num of the wav files (%d) not equals'
                  % (self.csv_file.shape[0], len(self.path2data_list)))

        self.dataset_names_train = []
        self.dataset_names_test = []
        self.labels_names_train = []
        self.labels_names_test = []
        self.run()

    def convert_csv_2_labels_values(self):
        for col in self.csv_file:
            labels_set_list = list(set(self.csv_file[col]))
            for i in range(self.csv_file[col].shape[0]):
                self.csv_file[col][i] = labels_set_list.index(self.csv_file[col][i])

    def convert_csv_to_labels_files(self):
        for i, row in self.csv_file.iterrows():
            name = str(row[0])
            np.save(arr=row[1:], file=os.path.join(self.path2save_labels, name))

    def del_previous_labels(self):
        previous_files = listdir(self.path2save_labels)
        for file in previous_files:
            os.remove(file)

    def split_train_test(self):
        random = np.random.uniform(size=len(self.path2data_list, ))
        selector = np.where(random > 0.8, 1., 0.)
        for i in range(selector.shape[0]):
            wav_file = os.path.join(self.path2data, self.path2data_list[i])
            label_file = os.path.join(self.path2save_labels, self.path2data_list[i].replace('wav', 'npy'))
            if selector[i] == 1:
                self.dataset_names_train.append(wav_file)
                self.labels_names_train.append(label_file)
            else:
                self.dataset_names_test.append(wav_file)
                self.labels_names_test.append(label_file)

    def run(self):
        self.del_previous_labels()
        self.convert_csv_2_labels_values()
        self.split_train_test()


if __name__ == '__main__':
    path2data = "/home/shlomis/PycharmProjects/TiFGAN/data/Data_custom_synth/"
    path2csv = "/home/shlomis/PycharmProjects/TiFGAN/data/Data_custom_synth.csv"
    path2save_labels = '/home/moshelaufer/PycharmProjects/datasets/buffer_noy_dataset/'
    data = Dataset(path2data=path2data, path2csv=path2csv, path2save_labels=path2save_labels)
    a=0