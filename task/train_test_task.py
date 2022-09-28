import mlflow
import mlflow.keras
import tensorflow as tf
from hydra.utils import instantiate
# from tensorflow.data import AUTOTUNE
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

# from model.self_supervised_model import Wav2Vec
# from losses.diversity_loss import DiversityLoss
# from losses.contrastive_loss import ContrastiveLoss


class TrainTask:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model = instantiate(cfg.model)

        self.datasets_path = instantiate(cfg.dataset_class)

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        self.path2save_model = self.cfg.get('path2save_model')

        self.model = instantiate(cfg.model)
        self.loss = instantiate(cfg.losses)
        self.epochs = self.cfg.get('epochs')
        self.callbacks = instantiate(cfg.callbacks)
        self.optimizer = instantiate(cfg.optimizer)
        self.train_steps_per_epoch = self.cfg.get('train_steps_per_epoch')

        self.processor = instantiate(cfg.processor)
        self.batch_size = self.cfg.get('batch_size')

        self.results = instantiate(cfg.results)

    def run(self):

        if self.datasets_path.labels_names is None:
            X_train, X_test, _, y_test = train_test_split(self.datasets_path.dataset_names,
                                                          self.datasets_path.dataset_names,
                                                          test_size=0.2,
                                                          random_state=1)

            X_test, X_val, y_test, y_val = train_test_split(X_test,
                                                            y_test,
                                                            test_size=0.05,
                                                            random_state=1)

            self.train_dataset = tf.data.Dataset.from_tensor_slices(X_train)
            self.test_dataset = tf.data.Dataset.from_tensor_slices(X_test)
            self.val_dataset = tf.data.Dataset.from_tensor_slices(X_val)

        else:
            X_train, X_test, y_train, y_test = train_test_split(self.datasets_path.dataset_names,
                                                                self.datasets_path.labels_names,
                                                                test_size=0.2,
                                                                random_state=1)

            X_test, X_val, y_test, y_val = train_test_split(X_test,
                                                            y_test,
                                                            test_size=0.05,
                                                            random_state=1)

            self.train_dataset = tf.data.Dataset.from_tensor_slices(list(zip(X_train, y_train)))
            self.test_dataset = tf.data.Dataset.from_tensor_slices(list(zip(X_test, y_test)))
            self.val_dataset = tf.data.Dataset.from_tensor_slices(list(zip(X_val, y_val)))

        self.train_dataset = (self.train_dataset
                              .shuffle(1024)
                              .map(self.processor, num_parallel_calls=tf.data.AUTOTUNE)
                              .cache()
                              .repeat()
                              .batch(self.batch_size['train'])
                              .prefetch(tf.data.AUTOTUNE)
                              )

        self.test_dataset = (self.test_dataset
                             .shuffle(1024)
                             .map(self.processor, num_parallel_calls=tf.data.AUTOTUNE)
                             .cache()
                             .repeat()
                             .batch(self.batch_size['test'])
                             .prefetch(tf.data.AUTOTUNE)
                             )

        self.val_dataset = (self.val_dataset
                            .shuffle(1024)
                            .map(self.processor, num_parallel_calls=tf.data.AUTOTUNE)
                            .cache()
                            .repeat()
                            .batch(self.batch_size['valid'])
                            .prefetch(tf.data.AUTOTUNE)
                            )

        self.model.compile(optimizer="Adam", loss=self.loss)

        mlflow.keras.autolog()

        self.model.fit(x=self.train_dataset,
                       epochs=self.epochs,
                       verbose=1,
                       validation_data=self.val_dataset,
                       callbacks=self.callbacks,
                       steps_per_epoch=self.train_steps_per_epoch,
                       initial_epoch=0,
                       use_multiprocessing=True)

        tf.saved_model.save(self.model,
                            self.path2save_model)


if __name__ == '__main__':
    task = TrainTask(epochs=3, path2save_model=r"C:\Users\moshel\Desktop\cscscs",
                     path2load_dataset=r"C:\Users\moshel\Desktop\cscscs")
    task.run()

