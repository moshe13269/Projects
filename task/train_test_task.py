
import os
import utils
import mlflow
import mlflow.keras
import tensorflow as tf
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split


class TrainTestTask:

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
        self.input_shape = [tuple(lst) for lst in self.cfg.data2vec_train_task.TrainTask.get('input_shape')]

        self.processor = instantiate(cfg.data2vec_train_task.TrainTask.processor)
        self.processor.t_axis = utils.outputs_conv_size(cfg.data2vec_train_task.TrainTask.model.conv_encoder.conv_layers,
                                                  cfg.data2vec_train_task.TrainTask.model.conv_encoder.num_duplicate_layer,
                                                  # self.dataset_class.dataset_names_train[0], p=None, avg_pooling=True) #image
                                                  self.dataset_class.dataset_names_train[0], p=0, avg_pooling=False) #wav

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

        train_dataset = (self.train_dataset
                         .shuffle(1024)
                         .map(lambda item: tf.numpy_function(self.processor.load_data, [item], [tf.float32, tf.float32])
                              , num_parallel_calls=tf.data.AUTOTUNE).map(
            lambda x, y: ((x, y), y))  # map(tf.autograph.experimental.do_not_convert(lambda x, y: ((x, y), y)))
                         .cache()
                         # .repeat()
                         .batch(self.batch_size['train'])
                         .prefetch(tf.data.AUTOTUNE)
                         )

        test_dataset = (self.test_dataset
                        .shuffle(1024)
                        .map(lambda item: tf.numpy_function(self.processor.load_data, [item], [tf.float32, tf.float32])
                             , num_parallel_calls=tf.data.AUTOTUNE).map(lambda x, y: ((x, y), y))
                        .cache()
                        # .repeat()
                        .batch(self.batch_size['test'])
                        .prefetch(tf.data.AUTOTUNE)
                        )

        val_dataset = (self.val_dataset
                       .shuffle(1024)
                       .map(lambda item: tf.numpy_function(self.processor.load_data, [item], [tf.float32, tf.float32])
                            , num_parallel_calls=tf.data.AUTOTUNE).map(lambda x, y: ((x, y), y))
                       .cache()
                       # .repeat()
                       .batch(self.batch_size['valid'])
                       .prefetch(tf.data.AUTOTUNE)
                       )

        model = self.model.build()

        model.compile(optimizer=self.optimizer, loss=self.loss)

        mlflow.keras.autolog()

        # mlflow.keras.log_model(registered_model_name=self.model_name + str(datetime.datetime.now()),
        #                        log_models=True,
        #                        artifact_path='file:///C:/Users/moshe/PycharmProjects/mlflow',
        #                        keras_model=model)
        with tf.device('/device:GPU:0'):
            with mlflow.start_run():
                # log parameters
                # mlflow.log_param("hidden_layers", args.hidden_layers)
                # mlflow.log_param("output", args.output)
                mlflow.log_param("epochs", self.epochs)
                mlflow.log_param("loss_function", self.loss)
                # log metrics
                # mlflow.log_metric("binary_loss", ktrain_cls.get_binary_loss(history))
                # mlflow.log_metric("binary_acc", ktrain_cls.get_binary_acc(history))
                # mlflow.log_metric("validation_loss", ktrain_cls.get_binary_loss(history))
                # mlflow.log_metric("validation_acc", ktrain_cls.get_validation_acc(history))
                # mlflow.log_metric("average_loss", results[0])
                # mlflow.log_metric("average_acc", results[1])

                # log artifacts (matplotlib images for loss/accuracy)
                # mlflow.log_artifacts(r'C:\Users\moshe\PycharmProjects\mlflow\image')
                # log model
                # mlflow.keras.log_model(model, r'C:\Users\moshe\PycharmProjects\mlflow')



            # with mlflow.start_run():
                model.fit(x=train_dataset,
                          epochs=self.epochs,
                          verbose=1,
                          validation_data=val_dataset,
                          # callbacks=self.callbacks,
                          # steps_per_epoch=self.train_steps_per_epoch,
                          initial_epoch=0,
                          use_multiprocessing=True)

                folder_name = str(len([x[0] for x in os.walk(self.path2save_model)]) - 1)
                tf.keras.models.save_model(model=model, filepath=os.path.join(self.path2save_model, folder_name))
                mlflow.keras.save_model(model, os.path.join(self.path2save_model, folder_name, '_' + str(1)))


            # run_name = f'test_split_{test_inx}__val_split_{crossval_inx}'
            #
            # with self.logger.start_run(run_name=run_name, nested=True):

            # prediction = self.model.evaluate(x=self.test_dataset)

            # # option
            # results = self.results(prediction)
            # mlflow.log_artifact() # path: str
            # mlflow.log_image() # image:ndarray

            # with mlflow.start_run() as run:
            #     mlflow.keras.log_model(self.data2vec_train_task, "models")
            #     mlflow.log_param("num_trees", n_estimators)
            #     mlflow.log_param("maxdepth", max_depth)
            #     mlflow.log_param("max_feat", max_features)

            # model.save(self.path2save_model)


if __name__ == '__main__':
    task = TrainTask(epochs=3, path2save_model=r"C:\Users\moshel\Desktop\cscscs",
                     path2load_dataset=r"C:\Users\moshel\Desktop\cscscs")
    task.run()
