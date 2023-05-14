import torch
import torch.nn as nn
import Projects_torch.layers
from typing import List, Tuple


class SynthAutoEncoder(nn.Module):
    num_layers: int
    d_model: int
    nhead: int

    linear_classifier: Projects_torch.layers.LinearClassifier

    def __init__(self,
                 conv_encoder: Projects_torch.layers.ConvFeatureExtractionModel,
                 num_layers,
                 d_model,
                 nhead,
                 linear_classifier: Projects_torch.layers.LinearClassifier,
                 ):

        super(SynthAutoEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.linear_classifier = linear_classifier

        self.conv_encoder = conv_encoder

    def forward(self, inputs):

        outputs = self.conv_encoder(inputs)

        outputs = torch.permute(outputs, (0, 2, 1))

        encoder_outputs = self.transformer(outputs)

        outputs_params_list = self.linear_classifier(encoder_outputs)

        return outputs_params_list