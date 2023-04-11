import os
import sys
import argparse
import numpy as np
import pandas as pd


class LabelsConverter:

    def __init__(self, path2save, path2csv):

        self.path2save = path2save
        self.path2csv = path2csv
        self.df = pd.read_csv(self.path2csv)
        self.labels_list = None

    def create_labels_list(self):
        labels_list = []
        for col in self.df.columns:
            if col != 'wav_id':
                labels_list.append(list(set(self.df[col])))
        self.labels_list = labels_list

    def convert_csv_rows2npy(self):

        for row in range(self.df.shape[0]):
            label = self.df.iloc[row].to_numpy()[1:]
            for i in range(label.shape[0]):
                value = self.labels_list[i].index(label[i])
                label[i] = value
            np.save(file=os.path.join(self.path2save, str(row)), arr=label)
        print('Labels had been created')
        return


def main(args):
    labels_converter = LabelsConverter(path2csv=args.path2csv, path2save=args.path2save)
    labels_converter.create_labels_list()
    labels_converter.convert_csv_rows2npy()


if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--path2csv', required=False,
                        help='path2csv', default='s')
    parser.add_argument('--path2save', required=False,
                        help='path2save', default='s')

    args = parser.parse_args()
    main(args)
