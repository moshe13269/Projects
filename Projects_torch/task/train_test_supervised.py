import os
from Projects_torch import torch_utils
from Projects_torch import losses
import torch
import torch.nn as nn
import dataset
import mlflow
import mlflow.pytorch
from mlflow import MlflowClient
from omegaconf import DictConfig
from hydra.utils import instantiate
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


class TrainTestTaskSupervised:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # self.dataset_class = instantiate(cfg.train_task.TrainTask.dataset_class)

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        self.steps_per_epoch = self.cfg.train_task.TrainTask.get('steps_per_epoch')
        self.validation_steps = self.cfg.train_task.TrainTask.get('validation_steps')

        self.path2save_model = self.cfg.train_task.TrainTask.get('path2save_model')
        self.path2save_csv = self.cfg.train_task.TrainTask.get('path2save_csv')

        # self.to_metrics = self.cfg.train_task.TrainTask.get('to_metrics')
        self.num_ce_loss = self.cfg.train_task.TrainTask.get('num_ce_loss')
        self.num_outputs = self.cfg.train_task.TrainTask.get('num_outputs')
        self.outputs_dimension_per_outputs = \
            cfg.train_task.TrainTask.model.linear_classifier.outputs_dimension_per_outputs

        # self.metrics_ = instantiate(cfg.train_task.TrainTask.metrics)
        self.metrics_outputs_dimension_per_outputs = cfg.train_task.TrainTask.metrics.outputs_dimension_per_outputs

        self.loss_weights = None

        self.epoch = 0
        self.epochs = self.cfg.train_task.TrainTask.get('epochs')
        self.callbacks = instantiate(cfg.train_task.TrainTask.callbacks)

        self.batch_size = self.cfg.train_task.TrainTask.get('batch_size')
        self.dataloader_ = instantiate(cfg.train_task.TrainTask.dataloader)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataloader_,
            batch_size=self.batch_size['train'],
            shuffle=True,
            num_workers=6
        )
        self.lr = self.cfg.train_task.TrainTask.get('learning_rate')
        self.devices = torch.device(self.cfg.train_task.TrainTask.get('devices'))
        self.loss = None
        self.model = None
        self.optimizer = None
        self.instantiate_model(cfg=cfg, path2load_model=None)
        self.model_name = self.cfg.train_task.TrainTask.get('model_name')
        self.path2save_plot_model = self.cfg.train_task.TrainTask.get('path2save_plot_model')
        self.running_loss = {'loss_param': [], 'loss_stft': []}

        # if self.cfg.train_task.TrainTask.get('to_schedule'):
        #     self.schedule = instantiate(cfg.train_task.TrainTask.schedule)
        #     self.callbacks = list(self.callbacks) + \
        #                      [tf.keras.callbacks.LearningRateScheduler(self.schedule.__call__)]

    def instantiate_model(self, cfg, path2load_model):

        self.model = instantiate(cfg.train_task.TrainTask.model)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        if path2load_model is not None:
            checkpoint = torch.load(path2load_model)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.loss = checkpoint['loss'].cuda()
            #
            # model.eval()
            # # - or -
        else:
            self.loss = losses.losses_instantiate(self.num_ce_loss,
                                                  cfg.train_task.TrainTask.loss_ce,
                                                  list(self.outputs_dimension_per_outputs),
                                                  cfg.train_task.TrainTask.loss)
            for i in range(len(self.loss)):
                self.loss[i] = self.loss[i].cuda()
        self.model.train()  # model.train(mode=False)

    def save_model(self):
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_stft': self.loss[0],
            'loss_param': self.loss[1]
        }, self.path2save_model)

    def set_on_gpus(self, dataset2gpu=False):
        self.model.to('cuda')  # .to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        # self.model = DDP(self.model, device_ids=[0, 1, 2, 3])
        self.model = self.model.cuda() #nn.DataParallel(self.model, device_ids=[0, 1, 2, 3]).cuda()
        self.model.train()

        if dataset2gpu:
            self.test_dataset.to(self.devices)
            self.train_dataset.to(self.devices)
            self.val_dataset.to(self.devices)

    def train_model(self):

        # model = self.model
        for epoch in range(self.epochs):

            # self.dataloader.shuffle_()

            running_loss_parmas_counter = 0.0
            running_loss_stft_counter = 0.0

            num_steps = 0

            for i, data in enumerate(self.dataloader, 0):
                inputs, labels = data
                labels = labels.cuda()
                inputs0 = inputs[0].cuda()#.to(self.devices)
                inputs1 = inputs[1].cuda()#.to(self.devices)
                inputs = [inputs0, inputs1]
                self.optimizer.zero_grad()

                try:
                    pred_param, stft_rec = self.model(inputs)
                except RuntimeError:
                    a=0
                # pred_param, stft_rec = self.model(inputs)
                loss_param = self.loss[1](pred_param, labels)
                loss_stft = self.loss[0](stft_rec, inputs[0])

                loss_param.backward(retain_graph=True)
                loss_stft.backward()
                self.optimizer.step()

                running_loss_parmas_counter += loss_param.item()
                running_loss_stft_counter += loss_stft.item()

                num_steps += 1

                if num_steps > 0 and num_steps % 200:
                    print('loss_param: %f, loss_stft: %f'
                          % (running_loss_parmas_counter/num_steps, running_loss_stft_counter/num_steps))

            running_loss_parmas_counter = running_loss_parmas_counter / num_steps
            running_loss_stft_counter = running_loss_stft_counter / num_steps
            self.running_loss['loss_param'].append(running_loss_parmas_counter)
            self.running_loss['loss_stft'].append(running_loss_stft_counter)

            self.custom_checkpoints()
            self.epoch += 1

        self.save_model()

    def custom_checkpoints(self):
        if len(self.running_loss['loss_param']) >= 2:
            if self.running_loss['loss_param'][-1] < self.running_loss['loss_param'][-2]:
                model = self.model
                model = model.cpu().state_dict()
                torch.save({
                    'epoch': self.epoch,
                    'model_state_dict': model,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.loss,
                }, self.path2save_model)

    def run(self):
        # self.train_dataset, self.test_dataset, self.val_dataset = dataset.split_dataset(self.dataset_class)
        #
        # train_dataset = dataset.torch_data_loader(self.train_dataset,
        #                                           self.processor.load_data,
        #                                           self.batch_size['train'])
        #
        # test_dataset = dataset.torch_data_loader(self.test_dataset,
        #                                          self.processor.load_data,
        #                                          self.batch_size['test'])
        #
        # val_dataset = dataset.torch_data_loader(self.val_dataset,
        #                                         self.processor.load_data,
        #                                         self.batch_size['valid'])
        #
        # self.data_loader = {'train': train_dataset,
        #                     'test': test_dataset,
        #                     'valid': val_dataset}

        torch_utils.utils.save_plot_model(model=self.model) #, input_shape=None)

        mlflow.pytorch.autolog()
        with mlflow.start_run():  # as run:
            self.set_on_gpus()
            self.train_model()

            # self.results.csv_predicted(self.model, test_dataset)
