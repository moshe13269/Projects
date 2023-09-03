import os
import hydra
import argparse
from Projects_torch import task
from omegaconf import DictConfig
from dataset.csv2npy_noy import main_

"""
path2data: path2dataset/{train, test, valid}/{data, labels, .csv}
path2data: path2dataset/{train, test, valid}/{data, .csv}
path2data: path2dataset/{data, labels, .csv}
path2data: path2dataset/{data, .csv}

path2save: path to empty folder which all the outputs will be saved 

"""


@hydra.main(config_path=os.path.join('../config', 'synth_decoder_STFT_noy'), config_name='config')
def main(cfg: DictConfig) -> None:
    parser = argparse.ArgumentParser(description=None)

    parser.add_argument('--path2data', required=False,
                        default=r'C:\Users\moshe\PycharmProjects\commercial_synth_dataset\noy\data',
                        help='path2data')

    parser.add_argument('--path2save', required=False,
                        default=r'C:\Users\moshe\PycharmProjects\commercial_synth_dataset\outputs_main',
                        help='path2save')

    parser.add_argument('--batch', required=False,
                        default=2,
                        help='batch')

    args = parser.parse_args()

    task.TrainTaskSupervised(cfg, args)


if __name__ == "__main__":
    main()
