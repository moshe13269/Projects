import os
from Projects_tensorflow import losses
import dataset
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from hydra.utils import instantiate
from omegaconf import DictConfig
# from tensorflow.keras.utils.vis_utils import plot_model


class TrainTestTaskSupervised:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        self.dataset_class = instantiate(cfg.train_task.TrainTask.dataset_class)

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        self.steps_per_epoch = self.cfg.train_task.TrainTask.get('steps_per_epoch')
        self.validation_steps = self.cfg.train_task.TrainTask.get('validation_steps')

        self.path2save_model = self.cfg.train_task.TrainTask.get('path2save_model')
        self.path2save_csv = self.cfg.train_task.TrainTask.get('path2save_csv')

        self.to_metrics = self.cfg.train_task.TrainTask.get('to_metrics')
        self.num_ce_loss = self.cfg.train_task.TrainTask.get('num_ce_loss')
        self.num_outputs = self.cfg.train_task.TrainTask.get('num_outputs')
        self.outputs_dimension_per_outputs = \
            cfg.train_task.TrainTask.model.linear_classifier.outputs_dimension_per_outputs

        self.loss = losses.losses_instantiate(self.num_ce_loss,
                                              cfg,
                                              list(self.outputs_dimension_per_outputs))

        # self.metrics = metrics.acc_metrics_instantiate2(
        #     self.to_metrics,
        #     list(self.outputs_dimension_per_outputs),
        #     cfg.train_task.TrainTask.metrics)

        self.metrics_ = instantiate(cfg.train_task.TrainTask.metrics)
        self.metrics_outputs_dimension_per_outputs = cfg.train_task.TrainTask.metrics.outputs_dimension_per_outputs

        self.loss_weights = None
        self.optimizer = instantiate(cfg.train_task.TrainTask.optimizer)
        self.epochs = self.cfg.train_task.TrainTask.get('epochs')
        self.callbacks = instantiate(cfg.train_task.TrainTask.callbacks)

        self.model_name = self.cfg.train_task.TrainTask.get('model_name')
        path2saved_model = self.cfg.train_task.TrainTask.get('saved_model')
        if path2saved_model is not None:
            loaded_model = tf.keras.models.load_model(path2saved_model, compile=False)

            model = instantiate(cfg.train_task.TrainTask.model)
            self.model = model.build()
            self.model.set_weights(loaded_model.get_weights())
            opt_ = loaded_model.optimizer  # tf.keras.optimizers.Adam()
            adam = tf.keras.optimizers.Adam()
            adam.set_weights(opt_.get_weights())

            self.model.compile(optimizer=adam,  # 'adam',
                               loss=list(self.loss),
                               metrics=None,  # self.metrics_,
                               loss_weights=self.loss_weights)
            # self.model.compile(optimizer=self.model.optimizer.set_weights(m.optimizer))
            # self.model.optimizer.lr = 1.1e-6
            a = 0
        else:
            # print('create model!!!')
            model = instantiate(cfg.train_task.TrainTask.model)
            # print('create model!!!')
            self.model = model.build()
            self.model.compile(optimizer=self.optimizer,
                               loss=list(self.loss),
                               metrics=None,  # self.metrics_,
                               loss_weights=self.loss_weights)

        self.path2save_plot_model = self.cfg.train_task.TrainTask.get('path2save_plot_model')

        # self.optimizer = optimizer.crate_optimizers_list(model=self.model)

        self.processor = instantiate(cfg.train_task.TrainTask.processor)
        self.batch_size = self.cfg.train_task.TrainTask.get('batch_size')

        self.results = instantiate(cfg.train_task.TrainTask.results)

        if self.cfg.train_task.TrainTask.get('to_schedule'):
            self.schedule = instantiate(cfg.train_task.TrainTask.schedule)
            self.callbacks = list(self.callbacks) + \
                             [tf.keras.callbacks.LearningRateScheduler(self.schedule.__call__)]

    def run(self):
        self.train_dataset, self.test_dataset, self.val_dataset = dataset.split_dataset(self.dataset_class)

        train_dataset = dataset.data_loader(self.train_dataset,
                                            self.processor.load_data,
                                            self.processor.mask,
                                            self.num_outputs,
                                            self.batch_size['train'])

        test_dataset = dataset.data_loader(self.test_dataset,
                                           self.processor.load_data,
                                           self.processor.mask,
                                           self.num_outputs,
                                           self.batch_size['test'])

        val_dataset = dataset.data_loader(self.val_dataset,
                                          self.processor.load_data,
                                          self.processor.mask,
                                          self.num_outputs,
                                          self.batch_size['valid'])

        tf.keras.utils.plot_model(self.model,
                                  to_file=self.path2save_plot_model,
                                  show_shapes=True,
                                  show_layer_names=True)

        mlflow.keras.autolog()

        with tf.device('/GPU:2'):
            with mlflow.start_run():
                self.model.fit(x=train_dataset,
                               epochs=self.epochs,
                               verbose=1,
                               validation_data=val_dataset,
                               callbacks=self.callbacks,
                               steps_per_epoch=self.steps_per_epoch,
                               # validation_steps=self.validation_steps,
                               # initial_epoch=0,
                               # use_multiprocessing=True
                               )

                folder_name = str(len([x[0] for x in os.walk(self.path2save_model)]) - 1)
                mlflow.keras.save_model(self.model, os.path.join(self.path2save_model, folder_name))
                tf.keras.models.save_model(model=self.model,
                                           filepath=os.path.join(self.path2save_model, folder_name + 'model_ft'))

            self.results.csv_predicted(self.model, test_dataset)
