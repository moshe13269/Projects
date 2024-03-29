
import os
from Projects_tensorflow import task
import hydra
from omegaconf import DictConfig


@hydra.main(config_path=os.path.join('../config', 'inference_synth_tal'), config_name='config')
def main(cfg: DictConfig) -> None:
    inference = task.Inference(cfg)
    inference.run()


if __name__ == "__main__":
    main()
