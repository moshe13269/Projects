import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torchaudio
import torch.nn as nn
import matplotlib.image
from os.path import join
from ce_loss import CELoss
import lightning.pytorch as pl


class LitModelWav2Vec2(pl.LightningModule):
    def __init__(self, ce_loss, num_outputs, learn_rate):
        super().__init__()

        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.model = bundle.get_model()

        self.linear1 = nn.Linear(29, 1)
        self.linear2 = nn.Linear(50, num_outputs)
        self.loss = ce_loss
        self.learn_rate = learn_rate

    def forward(self, x):
        with torch.inference_mode():
            features, _ = self.model.extract_features(x)

        features = torch.tensor(features[0].numpy())

        features = torch.nn.functional.relu(self.linear1(features))
        features = torch.nn.functional.relu(self.linear2(torch.squeeze(features)))
        return features

    def training_step(self, batch, batch_idx):
        labels = batch[1]
        output = self(batch[0])

        # loss
        loss = 0.0
        loss_ = self.loss(output, labels)
        loss += loss_

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch[1]
        output = self(batch[0])

        # loss
        loss = 0.0
        loss_ = self.loss(output, labels)
        loss += loss_

        self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.learn_rate)
