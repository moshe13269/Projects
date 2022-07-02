
import tensorflow as tf
from processors.processor import Processor
from model.selfsupervised_model import Wav2Vec
from losses.diversity_loss import DiversityLoss
from losses.contrastive_loss import ContrastiveLoss


class TrainTask:

    def __init__(self, epochs, path2save_model, path2load_dataset):
        self.path2save_model = path2save_model
        self.path2load_dataset = path2load_dataset
        self.model = Wav2Vec()
        self.processor_train = Processor()
        self.processor_validation = Processor()
        self.loss = [ContrastiveLoss(), DiversityLoss()]
        self.epochs = epochs

    def run(self):
        self.model.compile(optimizer="Adam", loss=self.loss)
        self.model.fit(x=self.processor_train, epochs=self.epochs, verbose=1, validation_data=self.processor_validation,
                       initial_epoch=0, use_multiprocessing=True)
        tf.saved_model.save(self.model, self.path2save_model)


if __name__ == '__main__':
    task = TrainTask(epochs=3, path2save_model=r"C:\Users\moshel\Desktop\cscscs",
                     path2load_dataset=r"C:\Users\moshel\Desktop\cscscs")
    task.run()