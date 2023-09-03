import os
import numpy as np
import pandas as pd

path2labels = r'D:\dataset\DAX7\output'

names = ['train', 'test', 'valid']

labels_set = []

for name in names:
    path = os.path.join(path2labels, name, 'labels')

    if not os.path.exists(os.path.join(os.getcwd(), os.path.join(path2labels, name), 'filtered_labels')):
        folder2makedir = path.replace('labels', 'filtered_labels')
        os.makedirs(folder2makedir)

    files_list = os.listdir(path)
    labels_train = [np.expand_dims(np.load(os.path.join(path, file)), 0) for file in files_list]

    labels_train_con = np.concatenate(labels_train, axis=0)
    labels_set.append(labels_train_con)

labels_set_con = np.concatenate(labels_set, axis=0)
labels_set_con.compress(~np.all(labels_set_con == 0, axis=0), axis=1)

labels_set_a = [l.compress(~np.all(l == 0, axis=0), axis=1) for l in labels_set]
