import torch
import numpy as np
from omegaconf import DictConfig
from Projects_torch import model
from Projects_torch import losses
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from utils.utils import init_weight_model, load_model
from torch.utils.data.sampler import SubsetRandomSampler
# from torch.utils.data.distributed import DistributedSampler


class TrainTaskSupervised:

    def __init__(self, cfg: DictConfig, args):
        self.cfg = cfg

        ######################################
        # model_name & path2save
        ######################################
        self.path2save_model = self.cfg.train_task.TrainTask.get('path2save_model')
        self.model_name = self.cfg.train_task.TrainTask.get('model_name')
        self.path2load_model = self.cfg.train_task.TrainTask.get('path2load_model')

        ######################################
        # losses & learning_rate & epochs
        ######################################
        self.num_ce_loss = self.cfg.train_task.TrainTask.get('num_ce_loss')
        self.outputs_dimension_per_outputs = \
            cfg.train_task.TrainTask.model.linear_classifier.outputs_dimension_per_outputs

        self.epochs = self.cfg.train_task.TrainTask.get('epochs')

        self.learning_rate = self.cfg.train_task.TrainTask.get('learning_rate')

        self.loss = losses.losses_instantiate(self.num_ce_loss,
                                              cfg.train_task.TrainTask.loss_ce,
                                              list(self.outputs_dimension_per_outputs),
                                              cfg.train_task.TrainTask.loss)

        if not self.cfg.train_task.TrainTask.get('loss_l2'):
            self.loss = self.loss[1:]

        ################################
        # model instantiate
        ################################
        if self.path2load_model is None:
            self.model = instantiate(cfg.train_task.TrainTask.model)
            self.model.apply(init_weight_model)

            pl_model = model.pl_model.LitModel(model=self.model,
                                               losses=self.loss,
                                               learn_rate=self.learning_rate,
                                               logger=None)

            print("Learning rate: %f" % self.learning_rate)
        else:
            self.model, self.optimizer = load_model(self.path2load_model)

        ################################
        # dataset & dataloader
        ################################
        self.batch_size = self.cfg.train_task.TrainTask.get('batch_size')
        self.dataset = instantiate(cfg.train_task.TrainTask.processor)
        self.dataset.load_dataset(args.path2data)

        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.1 * dataset_size))
        # if shuffle_dataset:

        np.random.seed(123456)
        np.random.shuffle(indices)

        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(self.dataset,
                                                   batch_size=self.batch_size['train'],
                                                   sampler=train_sampler,
                                                   num_workers=1)

        validation_loader = torch.utils.data.DataLoader(self.dataset,
                                                        batch_size=self.batch_size['valid'],
                                                        sampler=valid_sampler,
                                                        num_workers=1)

        #######################################
        # logger & checkpoint & trainer & fit
        #######################################
        mlf_logger = MLFlowLogger(experiment_name=self.model_name,
                                  tracking_uri="file:./ml-runs",
                                  save_dir=None)

        checkpoint_callback = ModelCheckpoint(dirpath=self.path2save_model,
                                              save_weights_only=False,
                                              monitor='val_loss',
                                              save_last=True,
                                              verbose=True)

        if torch.cuda.is_available():
            accelerator = 'gpu'
        else:
            accelerator = 'cpu'
        print('Use the accelerator: {}'.format(accelerator))

        self.trainer = Trainer(logger=mlf_logger,
                               callbacks=[checkpoint_callback],
                               accelerator=accelerator,
                               max_epochs=self.epochs)

        self.trainer.fit(model=pl_model,
                         train_dataloaders=train_loader,
                         val_dataloaders=validation_loader,
                         ckpt_path=None)
