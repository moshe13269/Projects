import torch
import torchmetrics
import lightning.pytorch as pl


class LitModel(pl.LightningModule):
    def __init__(self, model, losses, learn_rate, logger, autoencoder=False):
        super().__init__()
        self.model = model
        self.log = logger
        self.losses = losses
        self.learn_rate = learn_rate
        self.autoencoder = autoencoder

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        loss = 0.0
        for i in range(len(self.losses)):
            loss += self.losses[i](output, y)

        acc = 0.0
        if type(output) == tuple and type(output[0]) == list:
            for output_ in output:
                if len(output_) > 2:

                    for i in range(len(y)):
                        acc += torchmetrics.functional.accuracy(output_[i], y[1][i].squeeze(),
                                                                task="multiclass",
                                                                num_classes=output_[i].shape[-1]) #accuracy(output_[i], y[1][i].squeeze())
                    acc = acc / len(y)
        else:
            for i in range(len(y)):
                acc += torchmetrics.functional.accuracy(output[i], y[i].squeeze(),
                                                        task="multiclass",
                                                        num_classes=output[i].shape[-1])
            acc = acc / len(y)
        # acc = accuracy(output, y)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        # self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        loss = 0.0
        for i in range(len(self.losses)):
            loss += self.losses[i](output, y)

        acc = 0.0
        if type(output) == tuple and type(output[0]) == list:
            for output_ in output:
                if len(output_) > 2:

                    for i in range(len(y)):
                        acc += torchmetrics.functional.accuracy(output_[i], y[1][i].squeeze(),
                                                                task="multiclass",
                                                                num_classes=output_[i].shape[
                                                                    -1])  # accuracy(output_[i], y[1][i].squeeze())
                    acc = acc / len(y)
        else:
            for i in range(len(y)):
                acc += torchmetrics.functional.accuracy(output[i], y[i].squeeze(),
                                                        task="multiclass",
                                                        num_classes=output[i].shape[-1])
            acc = acc / len(y)

        self.log("validation_loss", loss, on_epoch=True)
        self.log("validation_acc", acc, on_epoch=True)
        # self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learn_rate)
