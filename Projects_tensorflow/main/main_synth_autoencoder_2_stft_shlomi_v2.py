
import os
import sys
sys.path.append('/home/moshela/work/moshe/pycharm/Projects/')
sys.path.append('/home/moshela/work/moshe/pycharm/Projects/Projects_tensorflow')
from Projects_tensorflow import task
import hydra
from omegaconf import DictConfig


@hydra.main(config_path=os.path.join('../config', 'synth_autoencoder_2_STFT_shlomi_v2'),
            config_name='config')
def main(cfg: DictConfig) -> None:
    import argparse

    # sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--path2data', required=False,
                        help='path2data')
    parser.add_argument('--path2save', required=False,
                        help='path2save')

    args = parser.parse_args()

    train_task = task.TrainTestTaskSupervised(cfg, args)
    train_task.run()


if __name__ == "__main__":
    main()
