import os
import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
import tensorflow as tf
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split


class FineTuningTask:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        self.dataset_class = instantiate(cfg.train_task.TrainTask.dataset_class)

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        self.path2save_model = self.cfg.train_task.TrainTask.get('path2save_model')
        self.path2save_csv = self.cfg.train_task.TrainTask.get('path2save_csv')

        self.model_name = self.cfg.train_task.TrainTask.get('model_name')
        self.model = instantiate(cfg.train_task.TrainTask.model)
        self.loss = instantiate(cfg.train_task.TrainTask.loss)
        self.epochs = self.cfg.train_task.TrainTask.get('epochs')
        self.callbacks = instantiate(cfg.train_task.TrainTask.callbacks)
        self.optimizer = instantiate(cfg.train_task.TrainTask.optimizer)
        self.train_steps_per_epoch = self.cfg.train_task.TrainTask.get('train_steps_per_epoch')
        self.input_shape = [tuple(lst) for lst in self.cfg.train_task.TrainTask.get('input_shape')]

        self.processor = instantiate(cfg.train_task.TrainTask.processor)

        self.batch_size = self.cfg.train_task.TrainTask.get('batch_size')

        self.results = instantiate(cfg.train_task.TrainTask.results)

    def evaluate_model(self, model, test_set):

        num_sample = int(test_set.__len__().numpy())
        test_set = test_set.as_numpy_iterator()

        results = np.zeros((num_sample * 2, 16))

        for sample in range(num_sample):
            x, y = test_set.next()
            y_ = model.predict(x)
            results[2 * sample: 2 * sample + 1, :] = y_.squeeze()
            results[2 * sample + 1: 2 * sample + 2, :] = y.squeeze()
            # a = model.layers[2](model.layers[1](model.layers[0](x)))
            # a1 = tf.reduce_max(model.layers[2](model.layers[1](model.layers[0](x))), axis=-1)

        pd.DataFrame(results).to_csv(os.path.join(self.path2save_csv, 'csv_results.csv'))

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
                                                              test_size=0.1,
                                                              random_state=1)

            self.train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            self.test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
            self.val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

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

        train_dataset = (self.train_dataset
                         .shuffle(1024)
                         .map(
            lambda path2data, path2label: tf.numpy_function(self.processor.load_data, [(path2data, path2label)],
                                                            [tf.float32, tf.float32])
            , num_parallel_calls=tf.data.AUTOTUNE)
                         .cache()
                         .batch(self.batch_size['train'])
                         .prefetch(tf.data.AUTOTUNE)
                         )

        test_dataset = (self.test_dataset
                        .shuffle(1024)
                        .map(
            lambda path2data, path2label: tf.numpy_function(self.processor.load_data, [(path2data, path2label)],
                                                            [tf.float32, tf.float32])
            , num_parallel_calls=tf.data.AUTOTUNE)
                        .cache()
                        .batch(self.batch_size['test'])
                        .prefetch(tf.data.AUTOTUNE)
                        )

        val_dataset = (self.val_dataset
                       .shuffle(1024)
                       .map(
            lambda path2data, path2label: tf.numpy_function(self.processor.load_data, [(path2data, path2label)],
                                                            [tf.float32, tf.float32])
            , num_parallel_calls=tf.data.AUTOTUNE)
                       .cache()
                       .batch(self.batch_size['valid'])
                       .prefetch(tf.data.AUTOTUNE)
                       )

        model = self.model.build()

        model.compile(optimizer=self.optimizer, loss=self.loss)


        mlflow.keras.autolog()

        with tf.device('/GPU:0'):
            with mlflow.start_run():
                mlflow.log_param("epochs", self.epochs)
                mlflow.log_param("loss_function", self.loss)

                model.fit(x=train_dataset,
                          epochs=self.epochs,
                          verbose=1,
                          validation_data=val_dataset,
                          # steps_per_epoch=100,
                          # callbacks=self.callbacks,
                          # self.train_steps_per_epoch,
                          initial_epoch=0,
                          use_multiprocessing=True)

                # mlflow.keras.log_model(model, "file:///home/moshelaufer/PycharmProjects/mlflow/")

                self.evaluate_model(model, test_dataset)

                folder_name = str(len([x[0] for x in os.walk(self.path2save_model)]) - 1)
                mlflow.keras.save_model(model, os.path.join(self.path2save_model, folder_name))
                tf.keras.models.save_model(model=model, filepath=os.path.join(self.path2save_model, folder_name +
                                                                              'model_ft'))
