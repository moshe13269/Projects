import mlflow
import mlflow.keras
import datetime
import tensorflow as tf
from hydra.utils import instantiate
# from tensorflow.data import AUTOTUNE
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split


# from data2vec_train_task.self_supervised_model import Wav2Vec
# from losses.diversity_loss import DiversityLoss
# from losses.contrastive_loss import ContrastiveLoss


class TrainTask:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        self.dataset_class = instantiate(cfg.data2vec_train_task.TrainTask.dataset_class)

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        self.path2save_model = self.cfg.data2vec_train_task.TrainTask.get('path2save_model')

        self.model_name = self.cfg.data2vec_train_task.TrainTask.get('model_name')
        self.model = instantiate(cfg.data2vec_train_task.TrainTask.model)
        self.loss = instantiate(cfg.data2vec_train_task.TrainTask.loss)
        self.epochs = self.cfg.data2vec_train_task.TrainTask.get('epochs')
        self.callbacks = instantiate(cfg.data2vec_train_task.TrainTask.callbacks)
        self.optimizer = instantiate(cfg.data2vec_train_task.TrainTask.optimizer)
        self.train_steps_per_epoch = self.cfg.data2vec_train_task.TrainTask.get('train_steps_per_epoch')

        self.processor = instantiate(cfg.data2vec_train_task.TrainTask.processor)
        self.batch_size = self.cfg.data2vec_train_task.TrainTask.get('batch_size')

        self.results = instantiate(cfg.data2vec_train_task.TrainTask.results)

    def run(self):

        dataset_had_split = len(self.dataset_class.dataset_names_train) > 0 and \
                            len(self.dataset_class.dataset_names_test) > 0

        if self.dataset_class.labels:
            if dataset_had_split:
                X_train = self.dataset_class.dataset_names_train
                y_train = self.dataset_class.labels_names_train
                X_test = self.dataset_class.dataset_names_test
                y_test = self.dataset_class.labels_names_test
            else:
                X_train, X_test, y_train, y_test = train_test_split(self.dataset_class.dataset_names,
                                                                    self.dataset_class.labels_names,
                                                                    test_size=0.2,
                                                                    random_state=1)

            X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                              y_train,
                                                              test_size=0.05,
                                                              random_state=1)

            self.train_dataset = tf.data.Dataset.from_tensor_slices(list(zip(X_train, y_train)))
            self.test_dataset = tf.data.Dataset.from_tensor_slices(list(zip(X_test, y_test)))
            self.val_dataset = tf.data.Dataset.from_tensor_slices(list(zip(X_val, y_val)))

        else:
            if dataset_had_split:
                X_train = self.dataset_class.dataset_names_train
                X_test = self.dataset_class.dataset_names_test
            else:
                X_train, X_test, y_train, y_test = train_test_split(self.dataset_class.dataset_names_train,
                                                                    self.dataset_class.labels_names_train,
                                                                    test_size=0.2,
                                                                    random_state=1)

            X_test, X_val, y_test, y_val = train_test_split(X_test,
                                                            X_test,
                                                            test_size=0.05,
                                                            random_state=1)

            self.train_dataset = tf.data.Dataset.from_tensor_slices(X_train)
            self.test_dataset = tf.data.Dataset.from_tensor_slices(X_test)
            self.val_dataset = tf.data.Dataset.from_tensor_slices(X_val)

        self.train_dataset = (self.train_dataset
                              .shuffle(1024)
                              .map(self.processor.load_data, num_parallel_calls=tf.data.AUTOTUNE)
                              .cache()
                              .repeat()
                              .batch(self.batch_size['train'])
                              .prefetch(tf.data.AUTOTUNE)
                              )

        self.test_dataset = (self.test_dataset
                             .shuffle(1024)
                             .map(self.processor.load_data, num_parallel_calls=tf.data.AUTOTUNE)
                             .cache()
                             .repeat()
                             .batch(self.batch_size['test'])
                             .prefetch(tf.data.AUTOTUNE)
                             )

        self.val_dataset = (self.val_dataset
                            .shuffle(1024)
                            .map(self.processor.load_data, num_parallel_calls=tf.data.AUTOTUNE)
                            .cache()
                            .repeat()
                            .batch(self.batch_size['valid'])
                            .prefetch(tf.data.AUTOTUNE)
                            )

        self.model.compile(optimizer="Adam", loss=self.loss)

        mlflow.keras.autolog(registered_model_name=self.model_name + str(datetime.datetime.now()))

        self.model.fit(x=self.train_dataset,
                       epochs=self.epochs,
                       verbose=1,
                       validation_data=self.val_dataset,
                       callbacks=self.callbacks,
                       steps_per_epoch=self.train_steps_per_epoch,
                       initial_epoch=0,
                       use_multiprocessing=True)

        # run_name = f'test_split_{test_inx}__val_split_{crossval_inx}'
        #
        # with self.logger.start_run(run_name=run_name, nested=True):

        prediction = self.model.evaluate(x=self.test_dataset)

        # # option
        # results = self.results(prediction)
        # mlflow.log_artifact() # path: str
        # mlflow.log_image() # image:ndarray

        # with mlflow.start_run() as run:
        #     mlflow.keras.log_model(self.data2vec_train_task, "models")
        #     mlflow.log_param("num_trees", n_estimators)
        #     mlflow.log_param("maxdepth", max_depth)
        #     mlflow.log_param("max_feat", max_features)

        tf.saved_model.save(self.model,
                            self.path2save_model, )


if __name__ == '__main__':
    task = TrainTask(epochs=3, path2save_model=r"C:\Users\moshel\Desktop\cscscs",
                     path2load_dataset=r"C:\Users\moshel\Desktop\cscscs")
    task.run()
