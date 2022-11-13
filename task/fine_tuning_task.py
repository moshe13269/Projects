import os
import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
import tensorflow as tf
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split


class FineTuningTask:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        self.dataset_class = instantiate(cfg.data2vec_train_task.TrainTask.dataset_class)

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        self.path2save_model = self.cfg.data2vec_train_task.TrainTask.get('path2save_model')
        self.path2save_csv = self.cfg.data2vec_train_task.TrainTask.get('path2save_csv')

        self.model_name = self.cfg.data2vec_train_task.TrainTask.get('model_name')
        self.model = instantiate(cfg.data2vec_train_task.TrainTask.model)
        self.loss = instantiate(cfg.data2vec_train_task.TrainTask.loss)
        self.epochs = self.cfg.data2vec_train_task.TrainTask.get('epochs')
        self.callbacks = instantiate(cfg.data2vec_train_task.TrainTask.callbacks)
        self.optimizer = instantiate(cfg.data2vec_train_task.TrainTask.optimizer)
        self.train_steps_per_epoch = self.cfg.data2vec_train_task.TrainTask.get('train_steps_per_epoch')
        self.input_shape = [tuple(lst) for lst in self.cfg.data2vec_train_task.TrainTask.get('input_shape')]

        self.processor = instantiate(cfg.data2vec_train_task.TrainTask.processor)
        # self.processor.t_axis = outputs_conv_size(cfg.data2vec_train_task.TrainTask.model.conv_encoder.conv_layers,
        #                                           cfg.data2vec_train_task.TrainTask.model.conv_encoder.num_duplicate_layer,
        #                                           # self.dataset_class.dataset_names_train[0], p=None, avg_pooling=True) #image
        #                                           self.dataset_class.dataset_names_train[0], p=0, avg_pooling=False) #wav

        self.batch_size = self.cfg.data2vec_train_task.TrainTask.get('batch_size')

        self.results = instantiate(cfg.data2vec_train_task.TrainTask.results)

    def test_model(self, model, test_dataset):

        with tf.device('/device:GPU:0'):
            len_dataset = int(test_dataset.__len__().numpy())
            results = np.zeros((len_dataset * 2, 16))

            test_dataset = test_dataset.as_numpy_iterator()

            for step in range(len_dataset):
                x, y = test_dataset.next()
                prediction = model.predict(x)
                results[step * 2] = prediction
                results[step * 2 + 1] = y
            pd.DataFrame(results).to_csv(os.path.join(self.path2save_csv, 'predicted_param.csv'))

        print("Model prediction had been saved")

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

            self.train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))  # (list(zip(X_train, y_train)))
            self.test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))  # (list(zip(X_test, y_test)))
            self.val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))  # (list(zip(X_val, y_val)))

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

        with tf.device('/device:GPU:0'):
            with mlflow.start_run():
                mlflow.log_param("epochs", self.epochs)
                mlflow.log_param("loss_function", self.loss)

                model.fit(x=train_dataset,
                          epochs=self.epochs,
                          verbose=1,
                          validation_data=val_dataset,
                          # callbacks=self.callbacks,
                          steps_per_epoch=5,
                          initial_epoch=0,
                          use_multiprocessing=True)

                self.test_model(model, val_dataset)

                folder_name = str(len([x[0] for x in os.walk(self.path2save_model)]) - 1)
                mlflow.keras.save_model(model, os.path.join(self.path2save_model, folder_name))
