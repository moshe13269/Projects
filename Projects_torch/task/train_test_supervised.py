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
from utils.utils import init_weight_model, load_model
from torch.utils.data.distributed import DistributedSampler


class TrainTestTaskSupervised:

    def __init__(self, cfg: DictConfig, args):
        self.cfg = cfg

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        self.path2save_model = self.cfg.train_task.TrainTask.get('path2save_model')

        self.num_ce_loss = self.cfg.train_task.TrainTask.get('num_ce_loss')
        self.num_outputs = self.cfg.train_task.TrainTask.get('num_outputs')
        self.outputs_dimension_per_outputs = \
            cfg.train_task.TrainTask.model.linear_classifier.outputs_dimension_per_outputs

        self.epoch_counter = 0
        self.epochs = self.cfg.train_task.TrainTask.get('epochs')

        self.batch_size = self.cfg.train_task.TrainTask.get('batch_size') #args.batch
        self.dataset = instantiate(cfg.train_task.TrainTask.processor)
        self.dataset.load_dataset(args.path2data)
        print(len(self.dataset))
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size['train'],
            shuffle=True,
            num_workers=10
        )

        self.lr = self.cfg.train_task.TrainTask.get('learning_rate')
        self.loss = None

        self.loss = losses.losses_instantiate(self.num_ce_loss,
                                              cfg.train_task.TrainTask.loss_ce,
                                              list(self.outputs_dimension_per_outputs),
                                              cfg.train_task.TrainTask.loss)
        if not self.cfg.train_task.TrainTask.get('loss_l2'):
            self.loss = self.loss[1:]
        for i in range(len(self.loss)):
            self.loss[i] = self.loss[i].cuda()

        self.model_name = self.cfg.train_task.TrainTask.get('model_name')
        self.path2load_model = self.cfg.train_task.TrainTask.get('path2load_model')

        if self.path2load_model is None:
            self.model = instantiate(cfg.train_task.TrainTask.model)
            # init_weight_model(self.model)
            self.model.apply(init_weight_model)
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
            print("Learning rate: %f" % self.lr)
        else:
            self.model, self.optimizer = load_model(self.path2load_model)

        if torch.cuda.is_available():
            self.model.cuda()

        self.running_loss = {'loss_param': [], 'loss_stft': []}

    def save_model(self):
        torch.save({
            'epoch': self.epoch_counter,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_stft': self.loss[0],
            'loss_param': self.loss[1]
        }, self.path2save_model)

    def train_model(self):
        mlflow.pytorch.autolog()
        with mlflow.start_run() as run:

            self.custom_checkpoints(flag=True)

            for epoch in range(self.epochs):
                print('Start epoch %d' % epoch)
                running_loss_parmas_counter = 0.0
                running_loss_stft_counter = 0.0

                num_steps = 0

                for step, (inputs, labels) in enumerate(self.dataloader): #, 1
                    if torch.cuda.is_available():
                        labels = labels.cuda()
                    if isinstance(inputs, list) or isinstance(inputs, tuple):
                        if torch.cuda.is_available():
                            inputs0 = inputs[0].cuda()
                            inputs1 = inputs[1].cuda()
                        inputs = [inputs0, inputs1]
                    else:
                        if torch.cuda.is_available():
                            inputs = inputs.cuda()
                    self.optimizer.zero_grad()
                    output = self.model(inputs)

                    if isinstance(output, list):
                        pred_param, stft_rec = output
                        loss_param = self.loss[1](pred_param, labels)
                        loss_stft = self.loss[0](stft_rec, inputs[0])
                        loss_param.backward(retain_graph=True)
                        loss_stft.backward()
                        self.optimizer.step()

                        running_loss_parmas_counter += loss_param.item()
                        running_loss_stft_counter += loss_stft.item()

                        num_steps += 1

                        if num_steps > 0 and num_steps % 1000 == 0:
                            print('loss_param: %f, loss_stft: %f'
                                  % (running_loss_parmas_counter / step, running_loss_stft_counter / step))

                    else:
                        pred_param = output

                        loss_param = self.loss[0](pred_param, labels)

                        loss_param.backward()
                        self.optimizer.step()

                        running_loss_parmas_counter += loss_param.item()

                        num_steps += 1

                        if num_steps > 0 and num_steps % 100 == 0:
                            print('loss_param: %f, %f'
                                  % (running_loss_parmas_counter/step, running_loss_parmas_counter))
                            print(self.running_loss['loss_param'])
                        if step == 100:
                            break;

                running_loss_parmas_counter = running_loss_parmas_counter / num_steps
                running_loss_stft_counter = running_loss_stft_counter / num_steps

                print('Loss param: %f' % running_loss_parmas_counter)
                self.running_loss['loss_param'].append(running_loss_parmas_counter)
                if len(self.loss) > 1:
                    self.running_loss['loss_stft'].append(running_loss_stft_counter)
                    print('Loss stft: %f' % running_loss_stft_counter)

                self.custom_checkpoints()
                self.epoch_counter += 1

        self.save_model()

    def custom_checkpoints(self, flag=False):
        if len(self.running_loss['loss_param']) >= 3:
            if (self.running_loss['loss_param'][-1] + 0.0001) < \
                    min(self.running_loss['loss_param'][:len(self.running_loss['loss_param'])-2]):
                print('sssdsdsds')
                model = self.model
                model = model.cpu().state_dict()
                torch.save({
                    'epoch': self.epoch_counter,
                    'model_state_dict': model,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss_param': self.running_loss['loss_param'],
                    'loss_stft': self.running_loss['loss_stft'],
                }, self.path2save_model)
                self.model.train().cuda()
                print('Model had been saved')
                print(self.running_loss['loss_param'], self.running_loss['loss_stft'])
        elif flag:
            model = self.model
            model = model.cpu().state_dict()
            torch.save({
                'epoch': self.epoch_counter,
                'model_state_dict': model,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss_param': self.running_loss['loss_param'],
                'loss_stft': self.running_loss['loss_stft'],
            }, self.path2save_model)
            self.model.train().cuda()
            print(self.running_loss['loss_param'], self.running_loss['loss_stft'])
    # def run(self):
    #     # self.train_dataset, self.test_dataset, self.val_dataset = dataset.split_dataset(self.dataset_class)
    #     #
    #     # train_dataset = dataset.torch_data_loader(self.train_dataset,
    #     #                                           self.processor.load_data,
    #     #                                           self.batch_size['train'])
    #     #
    #     # test_dataset = dataset.torch_data_loader(self.test_dataset,
    #     #                                          self.processor.load_data,
    #     #                                          self.batch_size['test'])
    #     #
    #     # val_dataset = dataset.torch_data_loader(self.val_dataset,
    #     #                                         self.processor.load_data,
    #     #                                         self.batch_size['valid'])
    #     #
    #     # self.data_loader = {'train': train_dataset,
    #     #                     'test': test_dataset,
    #     #                     'valid': val_dataset}
    #
    #     torch_utils.utils.save_plot_model(model=self.model) #, input_shape=None)
    #
    #     mlflow.pytorch.autolog()
    #     with mlflow.start_run():  # as run:
    #         self.set_on_gpus()
    #         self.train_model()

            # self.results.csv_predicted(self.model, test_dataset)



# def instantiate_model(self, cfg, path2load_model):
#
#     self.model = instantiate(cfg.train_task.TrainTask.model)
#     self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
#     if path2load_model is not None:
#         checkpoint = torch.load(path2load_model)
#         self.model.load_state_dict(checkpoint['model_state_dict'])
#         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         self.epoch = checkpoint['epoch']
#         self.loss = checkpoint['loss'].cuda()
#
#     else:
#         self.loss = losses.losses_instantiate(self.num_ce_loss,
#                                               cfg.train_task.TrainTask.loss_ce,
#                                               list(self.outputs_dimension_per_outputs),
#                                               cfg.train_task.TrainTask.loss)
#         for i in range(len(self.loss)):
#             self.loss[i] = self.loss[i].cuda()
#     self.model.train()  # model.train(mode=False)

# def set_on_gpus(self, dataset2gpu=False):
#     self.model.to('cuda')
#     self.model = self.model.cuda()
#
#     self.model.train()
#
#     if dataset2gpu:
#         self.test_dataset.to(self.devices)
#         self.train_dataset.to(self.devices)
#         self.val_dataset.to(self.devices)

