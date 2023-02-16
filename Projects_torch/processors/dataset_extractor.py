import os
import pickle
import numpy as np
import tensorflow as tf


class DownloadDadaset:
    path2save: str
    dataset_name: str

    def __init__(self,
                 path2save: str,
                 dataset_name: str):
        self.path2save = path2save
        self.dataset_name = dataset_name

    def _dataset_(self, dataset_):
        train_images = dataset_[0][0]
        train_labels = dataset_[0][1]
        test_images = dataset_[1][0]
        test_labels = dataset_[1][1]

        for i in range(len(train_images)):
            if not os.path.exists(os.path.join(self.path2save, 'train', 'data')):
                os.makedirs(os.path.join(self.path2save, 'train', 'data'))

            if not os.path.exists(os.path.join(self.path2save, 'train', 'labels')):
                os.makedirs(os.path.join(self.path2save, 'train', 'labels'))

            np.save(file=os.path.join(self.path2save, 'train', 'data', 'image' + '_' + str(i)),
                    arr=train_images[i] / 255.)

            np.save(file=os.path.join(self.path2save, 'train', 'labels', 'label' + '_' + str(i)),
                    arr=train_labels[i])

        for i in range(len(test_images)):
            if not os.path.exists(os.path.join(self.path2save, 'test', 'data')):
                os.makedirs(os.path.join(self.path2save, 'test', 'data'))

            if not os.path.exists(os.path.join(self.path2save, 'test', 'labels')):
                os.makedirs(os.path.join(self.path2save, 'test', 'labels'))

            np.save(file=os.path.join(self.path2save, 'test', 'data', 'image' + '_' + str(i)),
                    arr=test_images[i] / 255.)

            np.save(file=os.path.join(self.path2save, 'test', 'labels', 'label' + '_' + str(i)),
                    arr=test_labels[i])

        print('Dataset had been saved to: %s' % self.path2save)


if __name__ == '__main__':
    dd = DownloadDadaset(r'C:\Users\moshe\PycharmProjects\datasets', dataset_name='cifar10')
    dd._dataset_(tf.keras.datasets.cifar10.load_data())