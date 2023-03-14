
import os
from Projects_torch import task
import hydra
from omegaconf import DictConfig


@hydra.main(config_path=os.path.join('../config', 'data2vec_wav_ft'), config_name='config')
def main(cfg: DictConfig) -> None:
    train_task = task.FineTuningTask(cfg)
    train_task.run()


if __name__ == "__main__":
    main()
