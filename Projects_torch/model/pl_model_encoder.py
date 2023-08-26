import torch
import torch.nn as nn
import lightning.pytorch as pl


class LitModelEncoder(pl.LightningModule):
    def __init__(self, model, losses, learn_rate, outputs_dimension_per_outputs, num_classes):
        super().__init__()
        self.model = model.cuda()
        self.losses = losses
        self.learn_rate = learn_rate
        self.ce_loss = nn.CrossEntropyLoss()

        self.num_classes = num_classes

        self.outputs_dimension_per_outputs = outputs_dimension_per_outputs
        self.index2split = [sum(self.outputs_dimension_per_outputs[:i])
                            for i in range(len(self.outputs_dimension_per_outputs) + 1)]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        labels = batch[1]
        output = self(batch[0])

        loss_list = [0.0] * len(output)
        for i in range(len(output)):
            loss_list[i] += self.ce_loss(output[i], labels[:, i:i+1].squeeze())
            self.log("train_loss " + str(i), loss_list[i], on_step=True, on_epoch=True, prog_bar=True,
                     logger=True)

        loss = sum(loss_list) / len(loss_list)

        self.log("total train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch[1]
        output = self(batch[0])

        loss_list = [0.0] * len(output)
        for i in range(len(output)):
            loss_list[i] += self.ce_loss(output[i], labels[:, i:i+1].squeeze())
            self.log("validation_loss " + str(i), loss_list[i], on_step=True, on_epoch=True, prog_bar=True,
                     logger=True)

        loss = sum(loss_list) / len(loss_list)

        self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.learn_rate)


