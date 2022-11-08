import os
import hydra
from task.fine_tuning_task import TrainTask
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path=os.path.join('../config', 'data2vec_wav_ft'), config_name='config')
def main(cfg: DictConfig) -> None:
    train_task = TrainTask(cfg)
    train_task.run()


if __name__ == "__main__":
    main()
