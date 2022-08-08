
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


# class TrainValTestTask(BaseTask):
#
#     def __init__(self, cfg: DictConfig):
#         super().__init__()
#
#         self.cfg = cfg
#
#         self.verbose = True  # True/False
#         self.experiment_name = self.cfg.get("experiment_name", "train_val_test_task")
#
#         self.clustering = self.cfg.get('clustering') # True/False
#
#         self.dataset = instantiate(cfg.dataset)
#
#         print(len(self.dataset))
#         if self.cfg.get('filter_unknoun'):
#             for track in self.dataset.tracks:
#                 if track.track_meta['Label'] == 'Unknown':
#                     self.dataset.tracks.remove(track)
#             print(len(self.dataset))
#
#         # instantiate class lookup
#         self.class_lookup = None
#         if 'class_lookup' in cfg:
#             self.class_lookup = instantiate(cfg.class_lookup, _recursive_=False)
#
#         self.split = instantiate(cfg.split)
#         self.split.eval_test_crossval_splits(self.dataset, self.class_lookup)
#
#         # callbacks/losses/optimizer/result/batch_provider
#         self.callbacks = instantiate(cfg.callbacks)
#         self.losses = instantiate(cfg.losses)
#         self.metrics = instantiate(cfg.metrics)
#         self.optimizer = instantiate(cfg.optimizer)
#         self.batch_provider = instantiate(cfg.batch_provider)
#         self.logger = instantiate(cfg.logger)
#         self.result = instantiate(cfg.result, self.logger)
#
#     def run(self, upd_experiment_name=None):
#         super().run(upd_experiment_name)
#
#         # train/val/test loop
#         self.logger.set_experiment(self.experiment_name)
#
#         with self.logger.start_run(run_name='top_level'):
#
#             self.logger.log_config(self.cfg, "config.yaml")
#             self.logger.log_overrides()
#
#             for test_inx, test_data in enumerate(self.split.splits.items()):
#
#                 test_key, test_split = test_data
#                 test_info = test_split['test']
#
#                 for crossval_inx, train_val_info in enumerate(self.split.get_train_val_info(test_split, self.verbose)):
#
#                     run_name = f'test_split_{test_inx}__val_split_{crossval_inx}'
#
#                     with self.logger.start_run(run_name=run_name, nested=True):
#
#                         if self.verbose:
#                             print('running ' + run_name)
#
#                         train_info, val_info = train_val_info
#
#                         train_batch, train_steps_per_epoch = self.batch_provider.get_batch_provider(
#                             self.dataset, self.class_lookup, train_info, is_training=True)
#
#                         val_batch, val_steps_per_epoch = self.batch_provider.get_batch_provider(
#                             self.dataset, self.class_lookup, val_info)
#
#                         model_wrapper = instantiate(self.cfg.model)
#                         model = model_wrapper.model
#
#                         # tf.concat - contrastive loss, layer_5 - diveristy loss
#                         if len(self.losses) == 3:
#                             model.compile(loss=self.losses, optimizer=self.optimizer, loss_weights=[0.1, 0.5, 0.5])
#                         elif len(self.losses) == 2:
#                             model.compile(loss=self.losses, optimizer=self.optimizer, loss_weights=[0.1, 1.])
#                         else:
#                             model.compile(loss=self.losses, optimizer=self.optimizer, metrics=self.metrics)
#
#                         if not self.clustering:
#                             history = model.fit(train_batch,
#                                                 steps_per_epoch=train_steps_per_epoch,
#                                                 epochs=self.cfg.num_epochs,
#                                                 validation_data=val_batch,
#                                                 validation_steps=val_steps_per_epoch,
#                                                 callbacks=self.callbacks,
#                                                 initial_epoch=self.cfg.initial_epoch,
#                                                 verbose=self.verbose)  # , class_weight={0: 0.05, 1: 0.95})
