
import os
import hydra
from task.train_test_task import TrainTask
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path=os.path.join('../config', 'data2vec'), config_name='config')
def main(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    train_task = TrainTask(cfg)
    train_task.run()


if __name__ == "__main__":
    main()