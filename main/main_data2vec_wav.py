import os
import task
import hydra
from omegaconf import DictConfig, OmegaConf
from task import TrainTask



@hydra.main(config_path=os.path.join('../config', 'data2vec_wav'), config_name='config')
def main(cfg: DictConfig) -> None:
    train_task = task.TrainTask(cfg)
    train_task.run()


if __name__ == "__main__":
    # from tensorflow.python.client import device_lib
    #
    # print(device_lib.list_local_devices())
    main()
