import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torchaudio
import torch.nn as nn
import matplotlib.image
from os.path import join
from ce_loss import CELoss
import lightning.pytorch as pl
from transformers import Wav2Vec2ForCTC
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


class LitModelWav2Vec2(pl.LightningModule):
    def __init__(self, ce_loss, num_outputs, learn_rate):
        super().__init__()

        # bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        # self.model = bundle.get_model()
        # self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

        model_name = "facebook/wav2vec2-large-xlsr-53"
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)

        self.linear_model = nn.Sequential(
            nn.GELU(),
            nn.Linear(50, 1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(1024, num_outputs)
        )

        self.loss = ce_loss
        self.learn_rate = learn_rate

    def forward(self, x):
        i = self.feature_extractor(x, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            # o = self.model(i.input_values.cuda().squeeze())
            o = self.model(i.input_values.cuda().squeeze(), output_hidden_states=True)['hidden_states'][16]

        features = o
        # features = o.last_hidden_state
        # features = self.model(x, output_hidden_states=True)['hidden_states'][18]
        features = torch.transpose(features, 2, 1)
        features = self.linear_model(features)
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
        return torch.optim.AdamW(self.linear_model.parameters(), lr=self.learn_rate)
