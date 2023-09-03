import os
import torch
import numpy as np
import pandas as pd

path2folder = r'D:\dataset\DAX7'
file_list = [f for f in os.listdir(path=path2folder) if f.endswith('.pt')]

path2save = r'D:\dataset\DAX7\output'

names = ['train', 'test', 'valid']

for file in file_list:
    dir_ = os.path.join(path2folder, file)
    file_ = torch.load(dir_, map_location=torch.device('cpu'))
    stft, labels, config = file_.tensors

    stft = stft.numpy()
    stft_list = np.split(stft, stft.shape[0])
    stft_list = [np.squeeze(stft_) for stft_ in stft_list]

    labels = labels.numpy()
    labels_list = np.split(labels, labels.shape[0])
    labels_list = [np.squeeze(label_) for label_ in labels_list]

    config = config.numpy()
    config_list = np.split(config, config.shape[0])
    config_list = [np.squeeze(cfg_) for cfg_ in config_list]

    type_dataset = None
    for name in names:
        if name in file:
            type_dataset = name
    if type_dataset is None:
        type_dataset = 'train'

    os.makedirs(os.path.join(path2save, type_dataset, 'data'))
    os.makedirs(os.path.join(path2save, type_dataset, 'labels'))
    os.makedirs(os.path.join(path2save, type_dataset, 'config'))

    for i in range(len(stft)):
        file_name = os.path.join(path2save, type_dataset, 'data', str(i))
        np.save(arr=stft_list[i], file=file_name)

        file_name = os.path.join(path2save, type_dataset, 'labels', str(i))
        np.save(arr=labels[i], file=file_name)

        file_name = os.path.join(path2save, type_dataset, 'config', str(i))
        np.save(arr=config[i], file=file_name)

