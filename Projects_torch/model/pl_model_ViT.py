import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchmetrics import Accuracy


class LitModelEncoder(pl.LightningModule):
    def __init__(self, model, losses, learn_rate, outputs_dimension_per_outputs, num_classes):
        super().__init__()
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model
        self.losses = losses
        self.learn_rate = learn_rate

        self.num_classes = num_classes

        self.outputs_dimension_per_outputs = outputs_dimension_per_outputs
        self.index2split = [sum(self.outputs_dimension_per_outputs[:i])
                            for i in range(len(self.outputs_dimension_per_outputs) + 1)]
        self.ce_loss = [nn.CrossEntropyLoss() for _ in self.outputs_dimension_per_outputs]

        self.accuracies_list = [
            Accuracy(task="multiclass", num_classes=self.outputs_dimension_per_outputs[i], top_k=1)
            for i in range(len(self.outputs_dimension_per_outputs))]
        if torch.cuda.is_available():
            for acc in self.accuracies_list:
                acc.cuda()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        labels = batch[1]
        output = self(batch[0])

        output = [output[:, self.index2split[i]:self.index2split[i + 1]] for i in range(len(self.index2split) - 1)]

        loss = 0.0
        for i in range(len(output)):
            loss_i = self.ce_loss[i](output[i], labels[:, i:i + 1].squeeze())
            loss += loss_i
            self.log("train_loss " + str(i), loss_i, on_step=True, on_epoch=True, prog_bar=True,
                     logger=True)

        loss /= 9.

        self.log("total train_loss", loss, on_step=True, on_epoch=True, prog_bar=False,
                 logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch[1]
        output = self(batch[0])

        output = [output[:, self.index2split[i]:self.index2split[i + 1]] for i in range(len(self.index2split) - 1)]

        loss = 0.0
        accuracy = 0.0
        for i in range(len(output)):
            loss_i = self.ce_loss[i](output[i], labels[:, i:i + 1].squeeze())
            loss += loss_i
            self.log("validation_loss " + str(i), loss_i, on_step=True, on_epoch=True, prog_bar=False,
                     logger=True)

            acc_i = self.accuracies_list[i](output[i], labels[:, i:i + 1].squeeze())
            self.log("validation accuracy " + str(i), acc_i, on_step=True, on_epoch=True, prog_bar=False,
                     logger=True)

            accuracy += acc_i

        loss /= len(self.outputs_dimension_per_outputs)  # 9.
        accuracy /= len(self.outputs_dimension_per_outputs)

        self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=False,
                 logger=True)

        self.log("validation_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=False,
                 logger=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learn_rate, weight_decay=1e-5) #, betas=(0.5, 0.999))
