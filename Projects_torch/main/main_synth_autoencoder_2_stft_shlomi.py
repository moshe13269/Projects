
import os
import sys
sys.path.append('/home/moshela/work/moshe/pycharm/Projects/')
from Projects_torch import task
import hydra
from omegaconf import DictConfig


@hydra.main(config_path=os.path.join('../config', 'synth_autoencoder_2_STFT_shlomi'),
            config_name='config')
def main(cfg: DictConfig) -> None:
    train_task = task.TrainTestTaskSupervised(cfg)
    train_task.run()


if __name__ == "__main__":
    main()
