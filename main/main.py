
from task.train import TrainTask
from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    train_task = TrainTask()


if __name__ == "__main__":
    my_app()