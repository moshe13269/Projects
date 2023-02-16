import os
import pickle
import numpy as np
import pandas as pd


class LabelsConverter:

    class_data = []
    num_classes_refernce = []

    def float_labels2category_labels(self, data_frame, chosen_column_names):
        for name in chosen_column_names:
            num_classes = len(set(data_frame[name]))
            val_classes = list(set(data_frame[name]))
            val_classes.sort()
            self.num_classes_refernce.append(num_classes)
            self.class_data.append({'num_classes': num_classes, 'set': val_classes})

    def convert_float2classes(self, data_frame):
        data_frame = data_frame.to_numpy()
        for j in range(data_frame.shape[1]):
            for i in range(data_frame.shape[0]):
                data_frame[i, j] = self.class_data[j]['set'].index(data_frame[i, j])
        return data_frame

    @staticmethod
    def save_set(chosen_column, filtered_data_frame, path2save):
        lst_set = []
        for colum in chosen_column:
            lst_set.append(set(filtered_data_frame[colum]))
        with open(os.path.join(path2save, 'list_of_set_labels.pkl'), 'wb') as handle:
            pickle.dump(lst_set, handle)


def convert_str2float(data_frame):
    for col in data_frame.columns:
        labels = list(set(data_frame[col]))
        if type(labels[0]) == str:
            for i in range(data_frame.shape[0]):
                data_frame[col][i] = labels.index(data_frame[col][i])
    return data_frame


def choose_params_from_csv(path2csv):
    chosen_column = []
    data = pd.read_csv(path2csv)

    data = convert_str2float(data)
    columns_names = list(data.columns)

    for column in columns_names[1:]:
        column_values = data.loc[:, column]
        if len(set(column_values)) != 1: #column_values.min() != column_values.max():
            chosen_column.append(column)
    return chosen_column, data, data.loc[:, columns_names[0]]


def paras_labels(path2csv, path2save):

    labelsconverter = LabelsConverter()

    chosen_column, data_frame, files_names = choose_params_from_csv(path2csv)
    filtered_data_frame = data_frame.loc[:, chosen_column]
    labelsconverter.save_set(chosen_column, filtered_data_frame, path2save)
    labelsconverter.float_labels2category_labels(filtered_data_frame, chosen_column)
    filtered_data_frame = labelsconverter.convert_float2classes(filtered_data_frame)
    # filtered_data_frame = filtered_data_frame.to_numpy()
    files_names = list(files_names)

    for i in range(len(files_names)):
        label = filtered_data_frame[i, ]
        label = np.ndarray.astype(label, np.float32)
        np.save(arr=label, file=os.path.join(path2save, str(files_names[i])))
    np.save(arr=np.asarray(labelsconverter.num_classes_refernce), file=os.path.join(path2save, 'reference'))
    print('Labels files had been created')


# path2csv = "/home/shlomis/PycharmProjects/TiFGAN/data/Data_custom_synth.csv"
# path2save = '/home/moshelaufer/PycharmProjects/datasets/noy_synth/labels/'
# path2csv = "/home/moshelaufer/PycharmProjects/datasets/tal_noise/full_parameters.csv"
# path2save = "/home/moshelaufer/PycharmProjects/datasets/tal_noise/labels/"
path2csv = "/home/shlomis/PycharmProjects/Tal/full_parameters_osc_lfo_am.csv"
path2save = "/home/moshelaufer/PycharmProjects/datasets/tal_noise_shlomi/labels/"
paras_labels(path2csv, path2save)
