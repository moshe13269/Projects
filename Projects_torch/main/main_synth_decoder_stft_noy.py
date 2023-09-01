import os
import hydra
import argparse
from Projects_torch import task
from omegaconf import DictConfig
from dataset.csv2npy_noy import main_


@hydra.main(config_path=os.path.join('../config', 'synth_decoder_STFT_noy'), config_name='config')
def main(cfg: DictConfig) -> None:
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--path2data', required=False,
                        default=r'C:\Users\moshe\PycharmProjects\commercial_synth_dataset\noy\data',
                        # '/home/moshela/work/moshe/pycharm/dataset/noy/data/',
                        # r'C:\Users\moshe\PycharmProjects\commercial_synth_dataset\noy\data', #
                        help='path2data')
    # parser.add_argument('--path2save', required=False,
    #                     default=r'C:\Users\moshe\PycharmProjects\commercial_synth_dataset\outputs_main',
    #                     # '/home/moshela/work/moshe/pycharm/results',
    #                     # r'C:\Users\moshe\PycharmProjects\commercial_synth_dataset\outputs_main', #
    #                     help='path2save')

    parser.add_argument('--path2save_image', required=False,
                        default=None,
                        # '/home/moshela/work/moshe/pycharm/results',
                        # r'C:\Users\moshe\PycharmProjects\commercial_synth_dataset\outputs_main', #
                        help='path2save_image')

    parser.add_argument('--path2save_model', required=False,
                        default=None,
                        # '/home/moshela/work/moshe/pycharm/results',
                        # r'C:\Users\moshe\PycharmProjects\commercial_synth_dataset\outputs_main', #
                        help='path2save_model')

    parser.add_argument('--batch', required=False,
                        default=2,
                        help='batch')

    args = parser.parse_args()

    ################################
    # parser csv to labels
    ################################
    path2dataset = args.path2data

    if not os.path.exists(os.path.join(os.getcwd(), path2dataset, 'labels')):
        path2csv = [os.path.join(args.path2data, csv_file)
                    for csv_file in os.listdir(path2dataset) if csv_file.endswith('.csv')][0]

        os.makedirs(os.path.join(args.path2data, 'labels'))

        main_(path2csv=path2csv, path2save=os.path.join(args.path2data, 'labels'))

    task.TrainTaskSupervised(cfg, args)

    # train_task.train_model()


if __name__ == "__main__":
    main()
