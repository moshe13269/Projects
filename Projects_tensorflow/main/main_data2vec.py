
import os
from Projects_torch import task
import hydra
from omegaconf import DictConfig


@hydra.main(config_path=os.path.join('../config', 'data2vec'), config_name='config')
def main(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    train_task = task.TrainTask(cfg)
    train_task.run()


if __name__ == "__main__":
    main()