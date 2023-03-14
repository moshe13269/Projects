# import sys
#
# sys.path.append('../task/')
import os
from Projects_torch import task
import hydra
from omegaconf import DictConfig


@hydra.main(config_path=os.path.join('../config', 'ssl_synth_encoder'), config_name='config')
def main(cfg: DictConfig) -> None:
    train_task = task.TrainTestTask(cfg)
    train_task.run()


if __name__ == "__main__":
    main()
