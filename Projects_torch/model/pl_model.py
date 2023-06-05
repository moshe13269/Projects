import torch
import torch.nn as nn
import lightning.pytorch as pl
import torch.nn.functional as F


class LitModel(pl.LightningModule):
    def __init__(self, model, losses, learn_rate, logger):
        super().__init__()
        self.model = model
        self.log = logger
        self.losses = losses
        self.learn_rate = learn_rate

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        loss = 0.0
        for i in range(len(self.losses)):
            loss += self.losses[i](output[i], y[i])

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learn_rate)