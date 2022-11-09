
import os
import glob
import shutil
import numpy as np
from os import listdir
from scipy.io import wavfile
from os.path import isfile, join


def split_dataset(path2dataset, file_type, prob2choose_train, path2save):
    files_list = [f for f in listdir(path2dataset) if isfile(join(path2dataset, f)) and f.endswith(file_type)]

    arrange = np.arange(len(files_list))
    train_indexes = np.random.choice(len(files_list), int(prob2choose_train * len(files_list)), replace=False)
    test_indexes = [i for i in arrange if i not in train_indexes]

    for index in train_indexes:
        src = join(path2dataset, files_list[index])
        dst = join(path2save, 'train', 'data', files_list[index])
        shutil.copyfile(src, dst)

        src = join(path2dataset, files_list[index]).replace('audio_output', 'labels').replace('wav', 'npy')
        dst = join(path2save, 'train', 'labels', files_list[index].replace('wav', 'npy'))
        shutil.copyfile(src, dst)

    for index in test_indexes:
        src = join(path2dataset, files_list[index])
        dst = join(path2save, 'test', 'data', files_list[index])
        shutil.copyfile(src, dst)

        src = join(path2dataset, files_list[index]).replace('audio_output', 'labels').replace('wav', 'npy')
        dst = join(path2save, 'test', 'labels', files_list[index])
        shutil.copyfile(src, dst)
    print('The dataset had been created')


def del_files(path2datasets):
    list_of_path = [os.path.join(path2datasets, 'train', 'data', '*'),
                    os.path.join(path2datasets, 'train', 'labels', '*'),
                    os.path.join(path2datasets, 'test', 'data', '*'),
                    os.path.join(path2datasets, 'test', 'labels', '*')]

    for path in list_of_path:
        files = glob.glob(path)
        for f in files:
            os.remove(f)


def del_zeros(path2data):
    _, data = wavfile.read(path2data)
    if np.abs(np.max(data) - np.min(data)) <= 0.1:
        return 1
    return 0


def del_silence(path2dataset):
    files_list_train = [join(path2dataset, 'train', 'data', f) for f in
                        listdir(os.path.join(path2dataset, 'train', 'data'))]
    files_list_test = [join(path2dataset, 'test', 'data', f) for f in
                       listdir(os.path.join(path2dataset, 'test', 'data'))]
    counter = 0
    for file in files_list_train:
        if del_zeros(file):
            os.remove(file)
            file_label = file.replace('data\\', 'labels\\').replace('wav', 'npy')
            os.remove(file_label)
            counter += 1

    for file in files_list_test:
        if del_zeros(file):
            os.remove(file)
            file_label = file.replace('data\\', 'labels\\').replace('wav', 'npy')
            os.remove(file_label)
            counter += 1

    print('del %d' % counter)


path2dataset = r'C:\Users\moshe\PycharmProjects\commercial_synth_dataset\audio_output'
file_type = 'wav'
prob2choose_train = 0.9
path2save = r'C:\Users\moshe\PycharmProjects\datasets\tal_noise'
del_files(path2save)
split_dataset(path2dataset, file_type, prob2choose_train, path2save)
del_silence(path2save)




