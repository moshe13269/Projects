import os
from os import listdir
from os.path import isfile, join
import numpy as np

"""
1. dataset folder contained a two folders: dataset and labels
2. dataset are samples of dataset and labels are samples of labels
3. the names of the samples in the both folders are same
"""


class Dataset:
    path2load: str
    type_files: str

    def __init__(self, path2load, type_files):
        assert os.path.isdir(path2load), 'Directory not exist'

        self.path2load = path2load
        self.dataset_names = [file for file in listdir(join(path2load, 'dataset'))
                              if isfile(join(path2load, 'dataset', file)) and
                              os.path.splitext(file)[-1].lower() == type_files]

        self.labels_names = [file for file in listdir(join(path2load, 'labels'))
                             if isfile(join(path2load, 'labels', file)) and
                             os.path.splitext(file)[-1].lower() == type_files]

        def check_dataset():
            assert len(self.dataset_names) == len(self.labels_names), print('The number of dataset and labels ineqal')
            for index in range(len(self.dataset_names)):
                if self.dataset_names[index].replace('dataset', '') == self.labels_names[index].replace('labels', ''):

