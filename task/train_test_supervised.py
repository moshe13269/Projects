import os
import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
from copy import copy
import tensorflow as tf
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split


class TrainTestTaskSupervised:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        self.dataset_class = instantiate(cfg.train_task.TrainTask.dataset_class)

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        self.steps_per_epoch = self.cfg.train_task.TrainTask.get('steps_per_epoch')
        self.validation_steps = self.cfg.train_task.TrainTask.get('validation_steps')

        self.num_outputs = self.cfg.train_task.TrainTask.get('num_outputs')

        self.path2save_model = self.cfg.train_task.TrainTask.get('path2save_model')
        self.path2save_csv = self.cfg.train_task.TrainTask.get('path2save_csv')

        self.model_name = self.cfg.train_task.TrainTask.get('model_name')
        self.model = instantiate(cfg.train_task.TrainTask.model)

        def ce_loss_instantiate(outputs_dimension_per_outputs):
            loss_list = []
            for i in range(len(outputs_dimension_per_outputs)):
                loss = instantiate(cfg.train_task.TrainTask.loss_ce)
                loss.index_y_true = i
                loss.num_classes = outputs_dimension_per_outputs[i]
                loss.set_indexes()
                loss_list.append(loss)
            return loss_list

        self.loss_ce = instantiate(cfg.train_task.TrainTask.loss_ce)
            # ce_loss_instantiate(list(cfg.train_task.TrainTask.model.linear_classifier.outputs_dimension_per_outputs))

        if instantiate(cfg.train_task.TrainTask.loss)[0] is not None:
            self.loss = list(instantiate(cfg.train_task.TrainTask.loss)) + self.loss_ce
        else:
            self.loss = self.loss_ce

        def acc_metrics_instantiate(outputs_dimension_per_outputs):
            metrics_list = []
            for i in range(len(outputs_dimension_per_outputs)):
                metrics = instantiate(cfg.train_task.TrainTask.metrics)
                metrics.index_y_true = i
                metrics.num_classes = outputs_dimension_per_outputs[i]
                metrics.set_indexes()
                metrics_list.append(metrics)
            return metrics_list

        self.metrics = \
            acc_metrics_instantiate(list(cfg.train_task.TrainTask.model.linear_classifier.outputs_dimension_per_outputs))

        # [copy(loss) for i in range(self.cfg.train_task.TrainTask.get('num_outputs'))]
        self.epochs = self.cfg.train_task.TrainTask.get('epochs')
        self.callbacks = instantiate(cfg.train_task.TrainTask.callbacks)
        self.optimizer = instantiate(cfg.train_task.TrainTask.optimizer)

        self.processor = instantiate(cfg.train_task.TrainTask.processor)

        self.batch_size = self.cfg.train_task.TrainTask.get('batch_size')

        self.results = instantiate(cfg.train_task.TrainTask.results)

    def evaluate_model(self, model, test_set):

        num_sample = int(test_set.__len__().numpy())
        test_set = test_set.as_numpy_iterator()

        results = np.zeros((num_sample * 2, 16))

        for sample in range(num_sample):
            x, y = test_set.next()
            y_ = model.predict_on_batch(x)[0]  # model.predict(x)
            results[2 * sample: 2 * sample + 1, :] = y_.squeeze()
            results[2 * sample + 1: 2 * sample + 2, :] = y[0].squeeze()

        pd.DataFrame(results).to_csv(os.path.join(self.path2save_csv, 'csv_results.csv'))

    def run(self):

        dataset_had_split = len(self.dataset_class.dataset_names_train) > 0 and \
                            len(self.dataset_class.dataset_names_test) > 0

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

        train_dataset = (self.train_dataset
                         .shuffle(self.train_dataset.cardinality().numpy(), reshuffle_each_iteration=True)
                         .map(
            lambda path2data, path2label: tf.numpy_function(self.processor.load_data, [(path2data, path2label)],
                                                            [tf.float32, tf.float32])
            , num_parallel_calls=tf.data.AUTOTUNE) #.map(lambda x, y: (x, tuple([y for i in range(self.num_outputs)])))
                         .cache()
                         .batch(self.batch_size['train'])
                         .prefetch(tf.data.AUTOTUNE)
                         .repeat()
                         )

        test_dataset = (self.test_dataset
                        .shuffle(1024)
                        .map(
            lambda path2data, path2label: tf.numpy_function(self.processor.load_data, [(path2data, path2label)],
                                                            [tf.float32, tf.float32])
            , num_parallel_calls=tf.data.AUTOTUNE) #.map(lambda x, y: (x, tuple([y for i in range(self.num_outputs)])))
                        .cache()
                        .batch(self.batch_size['test'])
                        .prefetch(tf.data.AUTOTUNE)
                        )

        val_dataset = (self.val_dataset
                       .shuffle(self.val_dataset.cardinality().numpy(), reshuffle_each_iteration=True)
                       .map(
            lambda path2data, path2label: tf.numpy_function(self.processor.load_data, [(path2data, path2label)],
                                                            [tf.float32, tf.float32])
            , num_parallel_calls=tf.data.AUTOTUNE) #.map(lambda x, y: (x, tuple([y for i in range(self.num_outputs)])))
                       .cache()
                       .batch(self.batch_size['valid'])
                       .prefetch(tf.data.AUTOTUNE)
                       .repeat()
                       )

        model = self.model.build()
        plot_model(model, to_file='/home/moshelaufer/PycharmProjects/results/plot/model_plot.png', show_shapes=True,
                   show_layer_names=True)

        if type(self.loss) == list:

            if list(self.loss) == 19:
                loss_weights = {'concatenate': 1., 'concatenate_1': 0.1,
                                'concatenate_1_1': 1.,
                                'linear_classifier': 0.37, 'linear_classifier_1': 0.37, 'linear_classifier_2': 0.48,
                                'linear_classifier_3': 0.48, 'linear_classifier_4': 0.58, 'linear_classifier_5': 0.48,
                                'linear_classifier_6': 0.45, 'linear_classifier_7': 0.45, 'linear_classifier_8': 1.78,
                                'linear_classifier_9': 0.72, 'linear_classifier_10': 0.72, 'linear_classifier_11': 2.08,
                                'linear_classifier_12': 0.36, 'linear_classifier_13': 1.2, 'linear_classifier_14': 0.36,
                                'linear_classifier_15': 1.2}

            elif list(self.loss) == 15:
                loss_weights = {'linear_classifier': 0.37, 'linear_classifier_1': 0.37, 'linear_classifier_2': 0.48,
                                'linear_classifier_3': 0.48, 'linear_classifier_4': 0.58, 'linear_classifier_5': 0.48,
                                'linear_classifier_6': 0.45, 'linear_classifier_7': 0.45, 'linear_classifier_8': 1.78,
                                'linear_classifier_9': 0.72, 'linear_classifier_10': 0.72, 'linear_classifier_11': 2.08,
                                'linear_classifier_12': 0.36, 'linear_classifier_13': 1.2, 'linear_classifier_14': 0.36,
                                'linear_classifier_15': 1.2}

            metrics = {'linear_classifier': list(self.metrics)[0],
                        'linear_classifier_1': list(self.metrics)[1],
                        'linear_classifier_2': list(self.metrics)[2], 'linear_classifier_3': list(self.metrics)[3],
                        'linear_classifier_4': list(self.metrics)[4], 'linear_classifier_5': list(self.metrics)[5],
                        'linear_classifier_6': list(self.metrics)[6], 'linear_classifier_7': list(self.metrics)[7],
                        'linear_classifier_8': list(self.metrics)[8]}

        if type(self.loss) == list:
            model.compile(optimizer=self.optimizer,
                          loss=list(self.loss))
                      # metrics=metrics)#, loss_weights=loss_weights)
        else:
            model.compile(optimizer=self.optimizer,
                          loss=self.loss)

        mlflow.keras.autolog()

        with tf.device('/GPU:3'):
            with mlflow.start_run():
                # mlflow.keras.log_model(model, "models")
                # mlflow.log_param("epochs", self.epochs)
                # mlflow.log_param("loss_function", self.loss)
                # mlflow.log_param("epochs", self.epochs)
                # mlflow.log_param("learn_rate", tf.keras.backend.get_value(model.optimizer.learning_rate))

                model.fit(x=train_dataset,
                          epochs=self.epochs,
                          verbose=1,
                          validation_data=val_dataset,
                          callbacks=self.callbacks,
                          steps_per_epoch=self.steps_per_epoch,
                          validation_steps=self.validation_steps,
                          initial_epoch=0,
                          use_multiprocessing=True)

                # mlflow.keras.log_model(model, "file:///home/moshelaufer/PycharmProjects/mlflow/")

                # self.evaluate_model(model, test_dataset)

                folder_name = str(len([x[0] for x in os.walk(self.path2save_model)]) - 1)
                mlflow.keras.save_model(model, os.path.join(self.path2save_model, folder_name))
                tf.keras.models.save_model(model=model, filepath=os.path.join(self.path2save_model, folder_name +
                                                                              'model_ft'))
