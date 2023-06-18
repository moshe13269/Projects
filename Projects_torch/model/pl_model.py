import torch
import torchmetrics
import lightning.pytorch as pl


class LitModel(pl.LightningModule):
    def __init__(self, model, losses, learn_rate, autoencoder=False):
        super().__init__()
        self.model = model.cuda()
        self.losses = losses
        self.learn_rate = learn_rate
        self.autoencoder = autoencoder

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)

        # loss
        if len(self.losses) > 1:
            loss = 0.0
            for i in range(len(self.losses)):
                loss_ = self.losses[i](output, y)
                self.log("train_loss " + str(type(self.losses[i])), loss_, on_step=False, on_epoch=True, prog_bar=True,
                         logger=True)
                loss += loss_
        else:
            loss = 0.0
            for i in range(len(self.losses)):
                loss_ = self.losses[i](output, y)
                loss += loss_
                self.log("train_loss " + str(i), loss_, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # accuracy
        acc = 0.0
        if type(output) == tuple and type(output[0]) == list:
            for output_ in output:
                if len(output_) > 2:

                    for i in range(len(y)):
                        acc_ = torchmetrics.functional.accuracy(output_[i], y[1][i].squeeze(),
                                                                task="multiclass",
                                                                num_classes=output_[i].shape[-1]) #accuracy(output_[i], y[1][i].squeeze())
                        self.log("train_acc " + str(i), acc_, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                        acc += acc_
                    acc = acc / len(y)
        else:
            for i in range(len(y)):
                acc_ = torchmetrics.functional.accuracy(output[i], y[i].squeeze(),
                                                        task="multiclass",
                                                        num_classes=output[i].shape[-1])
                acc += acc_
                self.log("train_acc " + str(i), acc_, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            acc = acc / len(y)

        # acc = accuracy(output, y)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        loss = 0.0
        for i in range(len(self.losses)):
            loss_ = self.losses[i](output, y)
            loss += loss_
            self.log("valid_loss " + str(i), loss_, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        acc = 0.0
        if type(output) == tuple and type(output[0]) == list:
            for output_ in output:
                if len(output_) > 2:

                    for i in range(len(y)):
                        acc_ = torchmetrics.functional.accuracy(output_[i], y[1][i].squeeze(),
                                                                task="multiclass",
                                                                num_classes=output_[i].shape[
                                                                    -1])  # accuracy(output_[i], y[1][i].squeeze())
                        self.log("validation_acc " + str(i), acc, on_step=False, on_epoch=True, prog_bar=True,
                                 logger=True)
                        acc += acc_
                    acc = acc / len(y)
        else:
            for i in range(len(y)):
                acc_ = torchmetrics.functional.accuracy(output[i], y[i].squeeze(),
                                                        task="multiclass",
                                                        num_classes=output[i].shape[-1])
                self.log("validation_acc " + str(i), acc_, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                acc += acc_
            acc = acc / len(y)

        self.log("validation_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("validation_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.learn_rate)
