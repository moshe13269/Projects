import os
import sys
sys.path.append('/home/moshel/Projects/Projects_torch/')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import hydra
import argparse
# from Projects_torch.task import TrainTaskSupervised
from Projects_torch import task
# import Projects_torch.task

from omegaconf import DictConfig


@hydra.main(config_path=os.path.join('../config', 'synth_decoder_STFT_noy'), config_name='config')
def main(cfg: DictConfig) -> None:
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--path2data', required=False,
                        default=r'C:\Users\moshe\PycharmProjects\commercial_synth_dataset\noy\data',
                        # '/home/moshela/work/moshe/pycharm/dataset/noy/data/',
                        # r'C:\Users\moshe\PycharmProjects\commercial_synth_dataset\noy\data', #
                        help='path2data')
    parser.add_argument('--path2save', required=False,
                        default=r'C:\Users\moshe\PycharmProjects\commercial_synth_dataset\outputs_main',
                        # '/home/moshela/work/moshe/pycharm/results',
                        # r'C:\Users\moshe\PycharmProjects\commercial_synth_dataset\outputs_main', #
                        help='path2save')

    parser.add_argument('--batch', required=False,
                        default=2,
                        help='batch')

    args = parser.parse_args()

    task.TrainTaskSupervised(cfg, args)

    # train_task.train_model()


if __name__ == "__main__":
    main()
