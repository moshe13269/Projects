
import os
from Projects_torch import task
import hydra
from omegaconf import DictConfig


@hydra.main(config_path=os.path.join('../config_tensorflow', 'inference_synth_noy'), config_name='config_tensorflow')
def main(cfg: DictConfig) -> None:
    inference = task.Inference(cfg)
    inference.run()


if __name__ == "__main__":
    main()
