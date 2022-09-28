
import os
import sys
from os import listdir
from os.path import isfile, join
from dataclasses import dataclass

"""
1. dataset folder contained a two folders: dataset and labels
2. dataset are samples of dataset and labels are samples of labels
3. the names of the samples in the both folders are same
"""


@dataclass
class Dataset:
    path2load: str
    type_files: str
    labels: bool

    """class for load path to samples of dataset and the labels (if had a labels)"""

    def __init__(self, path2load, type_files, labels=False):
        assert os.path.isdir(path2load), 'Directory not exist'

        self.path2load = path2load
        self.dataset_names = [join(path2load, 'dataset', file) for file in listdir(join(path2load, 'dataset'))
                              if isfile(join(path2load, 'dataset', file)) and
                              os.path.splitext(file)[-1].lower() == type_files]
        if labels:
            self.labels_names = None
        else:
            self.labels_names = [join(path2load, 'labels', file) for file in listdir(join(path2load, 'labels'))
                                 if isfile(join(path2load, 'labels', file)) and
                                 os.path.splitext(file)[-1].lower() == type_files]

        if labels:
            def check_dataset():

                dataset_type_file = os.path.splitext(self.dataset_names[0])[-1].lower()
                label_type_file = os.path.splitext(self.labels_names[0])[-1].lower()

                assert len(self.dataset_names) == len(self.labels_names), \
                    print('The number of data samples and the labels is unequal')
                for index in range(len(self.dataset_names)):
                    if self.dataset_names[index].replace('dataset', '').replace(dataset_type_file, '') \
                            != self.labels_names[index].replace('labels', '').replace(label_type_file, ''):
                        sys.exit('The names of the files in the index: %d is not the same name in the dataset and the '
                                 'labels lists\n Dataset: %s\n Labels: %s'
                                 % (index, self.dataset_names[index], self.labels_names[index]))

            check_dataset()
            # self.dataset_names = list(zip(self.dataset_names, self.labels_names))
