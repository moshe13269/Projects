
import os
from Projects_torch import task
import hydra
from omegaconf import DictConfig


@hydra.main(config_path=os.path.join('../config', 'synth_encoder'), config_name='config')
def main(cfg: DictConfig) -> None:
    train_task = task.TrainTestTaskSupervised(cfg)
    train_task.run()


if __name__ == "__main__":
    main()
