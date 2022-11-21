import os
import numpy as np
import pandas as pd


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
    chosen_column, data_frame, files_names = choose_params_from_csv(path2csv)
    filtered_data_frame = data_frame.loc[:, chosen_column]
    filtered_data_frame = filtered_data_frame.to_numpy()
    filtered_data_frame = (filtered_data_frame - filtered_data_frame.min(axis=0)) / (
                filtered_data_frame.max(axis=0) - filtered_data_frame.min(axis=0))
    files_names = list(files_names)

    for i in range(len(files_names)):
        label = filtered_data_frame[i:i + 1, :]
        np.save(arr=label, file=os.path.join(path2save, str(files_names[i])))
    print('Labels files had been created')


path2csv = "/home/moshelaufer/PycharmProjects/datasets/tal_noise_25000/full_parameters.csv" #r"C:\Users\moshe\PycharmProjects\commercial_synth_dataset\full_parameters.csv"
path2save = "/home/moshelaufer/PycharmProjects/datasets/tal_noise_25000_labels/" # r"C:\Users\moshe\PycharmProjects\commercial_synth_dataset\labels"
paras_labels(path2csv, path2save)
