
import os
import task
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path=os.path.join('../config', 'inference_synth_noy'), config_name='config')
def main(cfg: DictConfig) -> None:
    inference = task.Inference(cfg)
    inference.run()


if __name__ == "__main__":
    main()
