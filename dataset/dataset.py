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
    dataset_names: str

    """class for load path to samples of dataset and the labels (if had a labels)"""

    def __init__(self, path2load, type_files, dataset_names, labels=False):
        assert os.path.isdir(path2load), 'Directory not exist'

        if dataset_names in path2load:
            self.path2load = path2load
        else:
            self.path2load = join(path2load, dataset_names)
        self.dataset_names = dataset_names
        self.type_files = type_files
        self.labels = labels

        self.dataset_names_train = [join(path2load, 'train', 'data', file)
                                    for file in listdir(join(path2load, 'train', 'data'))
                                    if isfile(join(path2load, 'train', 'data', file)) and
                                    os.path.splitext(file)[-1].lower()[1:] == self.type_files and
                                    'label' not in join(path2load, 'train', 'data', file)]

        self.dataset_names_test = [join(path2load, 'test', 'data', file)
                                   for file in listdir(join(path2load, 'test', 'data'))
                                   if isfile(join(path2load, 'test', 'data', file)) and
                                   os.path.splitext(file)[-1].lower()[1:] == self.type_files and
                                   'label' not in join(path2load, 'test', 'data', file)]
        if self.labels:
            # self.labels_names = None
            self.labels_names_train = [path.replace('wav', 'npy').replace(os.path.join('train', 'data'),
                                                                          os.path.join('train', 'labels'))      #.replace('data\\', 'labels\\')
                                       for path in self.dataset_names_train]

            self.labels_names_test = [path.replace('wav', 'npy').replace(os.path.join('test', 'data'),
                                                                         os.path.join('test', 'labels'))
                                       for path in self.dataset_names_test]
        else:
            self.labels_names_train = [join(path2load, 'labels', file)
                                       for file in listdir(join(path2load, 'train', 'labels'))
                                       if isfile(join(path2load, 'labels', file)) and
                                       os.path.splitext(file)[-1].lower()[1:] == self.type_files and
                                       'label' in join(path2load, 'dataset', file)]

            self.labels_names_test = [join(path2load, 'labels', file)
                                      for file in listdir(join(path2load, 'test', 'labels'))
                                      if isfile(join(path2load, 'labels', file)) and
                                      os.path.splitext(file)[-1].lower()[1:] == self.type_files and
                                      'label' in join(path2load, 'dataset', file)]

        # if self.labels:
        #     def check_dataset():
        #         # todo self.dataset_names take care where no had train - test
        #         dataset_type_file = os.path.splitext(self.dataset_names_train[0])[-1].lower()
        #         label_type_file = os.path.splitext(self.labels_names_train[0])[-1].lower()
        #
        #         assert len(self.dataset_names_train) == len(self.labels_names_train), \
        #             print('The number of data samples and the labels is unequal')
        #         for index in range(len(self.dataset_names_train)):
        #             if self.dataset_names_train[index].replace('dataset', '').replace(dataset_type_file, '') \
        #                     != self.labels_names_train[index].replace('labels', '').replace(label_type_file, ''):
        #                 sys.exit('The names of the files in the index: %d is not the same name in the dataset and the '
        #                          'labels lists\n Dataset: %s\n Labels: %s'
        #                          % (index, self.dataset_names_train[index], self.labels_names_train[index]))
        #
        #     check_dataset()
            # self.dataset_names = list(zip(self.dataset_names, self.labels_names))
