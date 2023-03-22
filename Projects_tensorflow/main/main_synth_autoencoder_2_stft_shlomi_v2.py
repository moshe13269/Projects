
import os
from Projects_tensorflow import task
import hydra
from omegaconf import DictConfig


@hydra.main(config_path=os.path.join('../config', 'synth_autoencoder_2_STFT_shlomi_v2'),
            config_name='config')
def main(cfg: DictConfig) -> None:
    train_task = task.TrainTestTaskSupervised(cfg)
    train_task.run()


if __name__ == "__main__":
    main()
