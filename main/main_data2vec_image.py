
import os
from Projects_torch import task
import hydra
from omegaconf import DictConfig


@hydra.main(config_path=os.path.join('../config_tensorflow', 'data2vec_image'), config_name='config_tensorflow')
def main(cfg: DictConfig) -> None:
    train_task = task.TrainTask(cfg)
    train_task.run()


if __name__ == "__main__":
    # from tensorflow.python.client import device_lib
    #
    # print(device_lib.list_local_devices())
    main()
