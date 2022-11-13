# import sys
#
# sys.path.append('../task/')
import os
import task
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path=os.path.join('../config', 'data2vec_wav'), config_name='config')
def main(cfg: DictConfig) -> None:
    train_task = task.TrainTestTask(cfg)
    train_task.run()


if __name__ == "__main__":
    main()
