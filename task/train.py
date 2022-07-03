
import tensorflow as tf
from hydra.utils import instantiate
from processors.processor import Processor
from omegaconf import DictConfig, OmegaConf
# from model.self_supervised_model import Wav2Vec
# from losses.diversity_loss import DiversityLoss
# from losses.contrastive_loss import ContrastiveLoss


class TrainTask:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.path2save_model = self.cfg.get('path2save_model')
        self.path2load_dataset = self.cfg.get('path2load_dataset')
        self.model = instantiate(cfg.model)
        self.processor_train = Processor()
        self.processor_validation = Processor()
        self.loss = instantiate(cfg.losses) #[ContrastiveLoss(), DiversityLoss()]
        self.epochs = self.cfg.get('epochs')

    def run(self):
        self.model.compile(optimizer="Adam", loss=self.loss)
        self.model.fit(x=self.processor_train, epochs=self.epochs, verbose=1, validation_data=self.processor_validation,
                       initial_epoch=0, use_multiprocessing=True)
        tf.saved_model.save(self.model, self.path2save_model)


if __name__ == '__main__':
    task = TrainTask(epochs=3, path2save_model=r"C:\Users\moshel\Desktop\cscscs",
                     path2load_dataset=r"C:\Users\moshel\Desktop\cscscs")
    task.run()