import torch
import numpy as np
import pandas as pd
from ce_loss import CELoss
from processor import DataLoader
from pl_model import LitModelWav2Vec2
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data.sampler import SubsetRandomSampler


def main(path2csv, dataset_path, path2save_model):
    np.random.seed(123456)

    epochs = 4
    df = pd.read_csv(path2csv)
    path2save_model = path2save_model
    batch_size = {'train': 64, 'test': 64}
    outputs_dimension_per_outputs = [len(set(df[key])) for key in df.keys()[1:]]

    #######################################################
    #   model
    #######################################################
    ce_loss = CELoss(outputs_dimension_per_outputs)
    model = LitModelWav2Vec2(ce_loss=ce_loss, num_outputs=sum(outputs_dimension_per_outputs), learn_rate=1e-3)

    #######################################################
    #   dataloader
    #######################################################
    dataloader = DataLoader(dataset_path=dataset_path)
    dataset_size = len(dataloader)
    indices = list(range(dataset_size))
    split = int(np.floor(0.5 * dataset_size))

    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataloader,
                                               batch_size=batch_size['train'],
                                               sampler=train_sampler,
                                               num_workers=1)

    test_loader = torch.utils.data.DataLoader(dataloader,
                                              batch_size=batch_size['test'],
                                              sampler=valid_sampler,
                                              num_workers=1)

    #######################################
    # logger & checkpoint & trainer & fit
    #######################################
    mlf_logger = MLFlowLogger(experiment_name='wav2vec2_noy',
                              tracking_uri="file:./ml-runs",
                              save_dir=None)

    checkpoint_callback = ModelCheckpoint(dirpath=path2save_model,
                                          save_weights_only=False,
                                          # monitor='val_loss',
                                          save_last=True,
                                          verbose=True)

    if torch.cuda.is_available():
        accelerator = 'gpu'
    else:
        accelerator = 'cpu'
    print('Use the accelerator: {}'.format(accelerator))

    trainer = Trainer(logger=mlf_logger,
                      callbacks=[checkpoint_callback],
                      accelerator=accelerator,
                      max_epochs=epochs)

    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=test_loader,
                ckpt_path=None)


if __name__ == '__main__':
    path2csv = r"C:\Users\moshe\PycharmProjects\commercial_synth_dataset\noy\Data_custom_synth.csv"
    dataset_path = r'C:\Users\moshe\PycharmProjects\commercial_synth_dataset\noy'
    path2save_model = r'C:\Users\moshe\PycharmProjects\commercial_synth_dataset'
    main(path2csv=path2csv, dataset_path=dataset_path, path2save_model=path2save_model)
