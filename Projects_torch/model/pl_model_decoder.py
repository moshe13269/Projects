# import os
import torch
import torch.nn as nn
import matplotlib.image
# from PIL import Image
from os.path import join
import lightning.pytorch as pl


class LitModelDecoder(pl.LightningModule):
    def __init__(self, model, losses, learn_rate, path2save_images):
        super().__init__()
        self.model = model.cuda()
        self.losses = losses
        self.learn_rate = learn_rate
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss(reduction='none')
        self.path2save_images = path2save_images

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        target_spec = batch[1]
        output = self(batch[0])

        # loss
        if len(self.losses) > 1:
            loss = 0.0
            for i in range(len(self.losses)):
                # loss_ = self.losses[i](output, target_spec)
                loss_ = torch.mean(torch.sum(torch.mean(self.l1_loss(output, target_spec), dim=-1), dim=-1))
                self.log("train_loss " + str(type(self.losses[i])), loss_, on_step=False, on_epoch=True, prog_bar=True,
                         logger=True)
                loss += loss_
        else:
            loss = 0.0
            loss_ = torch.mean(torch.sum(torch.mean(self.l1_loss(output, target_spec), dim=-1), dim=-1))
            # loss_ = self.mse_loss(output, target_spec) #self.losses[0](output, target_spec)
            loss += loss_
            # self.log("train_loss ", loss_, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        target_spec = batch[1]
        output = self(batch[0])

        if batch_idx == 1:
            self.save_images(output, target_spec)
        # loss
        if len(self.losses) > 1:
            loss = 0.0
            for i in range(len(self.losses)):
                loss_ = torch.mean(torch.sum(torch.mean(self.l1_loss(output, target_spec), dim=-1), dim=-1))
                # loss_ = self.losses[i](output, target_spec)
                self.log("valid_loss " + str(type(self.losses[i])), loss_, on_step=True, on_epoch=True, prog_bar=True,
                         logger=True)
                loss += loss_
        else:
            loss = 0.0
            loss_ = torch.mean(torch.sum(torch.mean(self.l1_loss(output, target_spec), dim=-1), dim=-1))
            # loss_ = self.mse_loss(output, target_spec) #self.losses[0](output, target_spec)
            loss += loss_
            # self.log("valid_loss ", loss_, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.learn_rate)

    def save_images(self, outputs, target_spec):
        target_list = torch.tensor_split(target_spec, target_spec.shape[0])
        target_list = [im.squeeze().cpu().numpy() for im in target_list]

        images_list = torch.tensor_split(outputs, outputs.shape[0])
        images_list = [im.squeeze().cpu().numpy() for im in images_list]
        for i in range(len(images_list)):
            matplotlib.image.imsave(join(self.path2save_images, str(i)+'.png'), images_list[i])
            matplotlib.image.imsave(join(self.path2save_images, str(i) + 'gt'+ '.png'), target_list[i])
            # im = Image.fromarray(images_list[i])
            # im.save(join(self.path2save_images, 'i.JPEG'))
