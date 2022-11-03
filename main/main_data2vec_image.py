import os
import hydra
from task.train_test_task import TrainTask
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path=os.path.join('../config', 'data2vec_image'), config_name='config')
def main(cfg: DictConfig) -> None:
    train_task = TrainTask(cfg)
    train_task.run()


if __name__ == "__main__":
    # from tensorflow.python.client import device_lib
    #
    # print(device_lib.list_local_devices())
    main()
