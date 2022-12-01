import os
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


def choose_params_from_csv(path2csv):
    chosen_column = []
    data = pd.read_csv(path2csv)
    columns_names = list(data.columns)

    for column in columns_names[1:]:
        column_values = data.loc[:, column]
        if column_values.min() != column_values.max():
            chosen_column.append(column)
    return chosen_column, data, data.loc[:, columns_names[0]]


def paras_labels(path2csv, path2save):

    labelsconverter = LabelsConverter()

    chosen_column, data_frame, files_names = choose_params_from_csv(path2csv)
    filtered_data_frame = data_frame.loc[:, chosen_column]
    labelsconverter.float_labels2category_labels(filtered_data_frame, chosen_column)
    filtered_data_frame = labelsconverter.convert_float2classes(filtered_data_frame)
    # filtered_data_frame = filtered_data_frame.to_numpy()
    files_names = list(files_names)

    for i in range(len(files_names)):
        label = filtered_data_frame[i, ]
        np.save(arr=label, file=os.path.join(path2save, str(files_names[i])))
    np.save(arr=np.asarray(labelsconverter.num_classes_refernce), file=os.path.join(path2save, 'reference'))
    print('Labels files had been created')


path2csv = "/home/moshelaufer/PycharmProjects/datasets/tal_noise_25000_base/full_parameters.csv" #r"C:\Users\moshe\PycharmProjects\commercial_synth_dataset\full_parameters.csv"
path2save = "/home/moshelaufer/PycharmProjects/datasets/tal_noise_25000_labels/" # r"C:\Users\moshe\PycharmProjects\commercial_synth_dataset\labels"
paras_labels(path2csv, path2save)
