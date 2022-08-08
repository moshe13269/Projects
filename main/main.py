
import os
import hydra
from task.train import TrainTask
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path=os.path.join('../config', 'wav2vec'), config_name='config')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    train_task = TrainTask()
    train_task.run()


if __name__ == "__main__":
    main()