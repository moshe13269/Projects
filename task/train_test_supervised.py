import os
import losses
import metrics
import dataset
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from keras.utils.vis_utils import plot_model


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
                                              cfg.train_task.TrainTask.loss_ce,
                                              list(self.outputs_dimension_per_outputs),
                                              cfg.train_task.TrainTask.loss)

        self.metrics = metrics.acc_metrics_instantiate(
            self.to_metrics,
            list(self.outputs_dimension_per_outputs),
            cfg.train_task.TrainTask.metrics)
        self.loss_weights = None
        self.optimizer = instantiate(cfg.train_task.TrainTask.optimizer)
        self.epochs = self.cfg.train_task.TrainTask.get('epochs')
        self.callbacks = instantiate(cfg.train_task.TrainTask.callbacks)

        self.model_name = self.cfg.train_task.TrainTask.get('model_name')
        path2saved_model = self.cfg.train_task.TrainTask.get('saved_model')
        if path2saved_model is not None:
            self.model = tf.keras.models.load_model(path2saved_model, compile=False)
            opt = tf.keras.optimizers.Adam()
            m = self.model
            self.model.compile(optimizer='adam',
                               loss=list(self.loss),
                               metrics=self.metrics,
                               loss_weights=self.loss_weights)
            self.model.compile(optimizer=self.model.optimizer.set_weights(m.optimizer))
            # self.model.optimizer.lr = 1.1e-6
        else:
            model = instantiate(cfg.train_task.TrainTask.model)
            self.model = model.build()
            self.model.compile(optimizer=self.optimizer,
                               loss=list(self.loss),
                               metrics=self.metrics,
                               loss_weights=self.loss_weights)

        self.path2save_plot_model = self.cfg.train_task.TrainTask.get('path2save_plot_model')




        # self.optimizer = optimizer.crate_optimizers_list(model=self.model)

        self.processor = instantiate(cfg.train_task.TrainTask.processor)
        self.batch_size = self.cfg.train_task.TrainTask.get('batch_size')

        self.results = instantiate(cfg.train_task.TrainTask.results)

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

        plot_model(self.model,
                   to_file=self.path2save_plot_model,
                   show_shapes=True,
                   show_layer_names=True)

        # model = load_model(f"model.h5")
        # old_weights = model.optimzier.get_weights()
        # new_opitmizer = horovod.DistributedOptimizer(...)
        # model.compile(optimizer=new_opitmizer)
        #
        # # use fake gradient to let optimizer init the wieghts.
        # grad_vars = model.trainable_weights
        # zero_grads = [tf.zeros_like(w) for w in grad_vars]
        # model.optimizer.apply_gradients(zip(zero_grads, grad_vars))
        #
        # # This will work because the new optimizer is initialized.
        # model.optimzier.set_weights(old_weights)

        mlflow.keras.autolog()

        with tf.device('/GPU:3'):
            with mlflow.start_run():
                self.model.fit(x=train_dataset,
                               epochs=self.epochs,
                               verbose=1,
                               validation_data=val_dataset,
                               callbacks=self.callbacks,
                               steps_per_epoch=self.steps_per_epoch,
                               validation_steps=self.validation_steps,
                               initial_epoch=0,
                               use_multiprocessing=True)

                folder_name = str(len([x[0] for x in os.walk(self.path2save_model)]) - 1)
                mlflow.keras.save_model(self.model, os.path.join(self.path2save_model, folder_name))
                tf.keras.models.save_model(model=self.model,
                                           filepath=os.path.join(self.path2save_model, folder_name + 'model_ft'))

            self.results.csv_predicted(self.model, test_dataset)

# mlflow.keras.log_model(model, "file:///home/moshelaufer/PycharmProjects/mlflow/")
# mlflow.keras.log_model(model, "models")
# mlflow.log_param("epochs", self.epochs)
# mlflow.log_param("loss_function", self.loss)
# mlflow.log_param("epochs", self.epochs)
# mlflow.log_param("learn_rate", tf.keras.backend.get_value(model.optimizer.learning_rate))

# if list(self.loss) == 19:
#            loss_weights = {'concatenate': 1., 'concatenate_1': 0.1,
#                            'concatenate_1_1': 1.,
#                            'linear_classifier': 0.37, 'linear_classifier_1': 0.37, 'linear_classifier_2': 0.48,
#                            'linear_classifier_3': 0.48, 'linear_classifier_4': 0.58, 'linear_classifier_5': 0.48,
#                            'linear_classifier_6': 0.45, 'linear_classifier_7': 0.45, 'linear_classifier_8': 1.78,
#                            'linear_classifier_9': 0.72, 'linear_classifier_10': 0.72, 'linear_classifier_11': 2.08,
#                            'linear_classifier_12': 0.36, 'linear_classifier_13': 1.2, 'linear_classifier_14': 0.36,
#                            'linear_classifier_15': 1.2}
#
#        elif list(self.loss) == 15:
#            loss_weights = {'linear_classifier': 0.37, 'linear_classifier_1': 0.37, 'linear_classifier_2': 0.48,
#                            'linear_classifier_3': 0.48, 'linear_classifier_4': 0.58, 'linear_classifier_5': 0.48,
#                            'linear_classifier_6': 0.45, 'linear_classifier_7': 0.45, 'linear_classifier_8': 1.78,
#                            'linear_classifier_9': 0.72, 'linear_classifier_10': 0.72, 'linear_classifier_11': 2.08,
#                            'linear_classifier_12': 0.36, 'linear_classifier_13': 1.2, 'linear_classifier_14': 0.36,
#                            'linear_classifier_15': 1.2}
#
#        elif list(self.loss) == 4:
#            loss_weights = {'linear_classifier': 1., 'concatenate': 0.1, 'concatenate_1': 0.1,
#                            'concatenate_1_1': 0.1}
#        elif list(self.loss) == 4:
#            loss_weights = {'linear_classifier': 1., 'wavs': 0.1, 'latent': 0.1}
#        else:
#            loss_weights = None
#
#        if self.to_metrics and type(self.loss_ce) == list and len(self.loss_ce) > 3:
#            metrics = {'linear_classifier': list(self.metrics)[0],
#                       'linear_classifier_1': list(self.metrics)[1],
#                       'linear_classifier_2': list(self.metrics)[2], 'linear_classifier_3': list(self.metrics)[3],
#                       'linear_classifier_4': list(self.metrics)[4], 'linear_classifier_5': list(self.metrics)[5],
#                       'linear_classifier_6': list(self.metrics)[6], 'linear_classifier_7': list(self.metrics)[7],
#                       'linear_classifier_8': list(self.metrics)[8]}
#        else:
#            metrics = {'linear_classifier': list(self.metrics)[0]}


# def ce_loss_instantiate(outputs_dimension_per_outputs):
#     loss_list = []
#     for i in range(len(outputs_dimension_per_outputs)):
#         loss = instantiate(cfg.train_task.TrainTask.loss_ce)
#         loss.index_y_true = i
#         loss.num_classes = outputs_dimension_per_outputs[i]
#         loss.set_indexes()
#         loss_list.append(loss)
#     return loss_list
#
# self.loss_ce = instantiate(cfg.train_task.TrainTask.loss_ce)  # ce_loss_instantiate( #
# # list(cfg.train_task.TrainTask.model.linear_classifier.outputs_dimension_per_outputs))
#
# if instantiate(cfg.train_task.TrainTask.loss)[0] is not None:
#     if type(self.loss_ce) == list:
#         self.loss = list(instantiate(cfg.train_task.TrainTask.loss)) + self.loss_ce
#     else:
#         self.loss = list(instantiate(cfg.train_task.TrainTask.loss)) + [self.loss_ce]
# else:
#     self.loss = self.loss_ce
