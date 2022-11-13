
import os
import task
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path=os.path.join('../config', 'data2vec_wav_ft'), config_name='config')
def main(cfg: DictConfig) -> None:
    train_task = task.TrainTask(cfg)
    train_task.run()


if __name__ == "__main__":
    main()
