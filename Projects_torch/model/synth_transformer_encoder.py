import torch
import torch.nn as nn
import Projects_torch.layers
from typing import List, Tuple


class SynthTransformerEncoder(nn.Module):
    num_layers: int
    d_model: int
    nhead: int

    linear_classifier: Projects_torch.layers.LinearClassifier

    def __init__(self,
                 conv_encoder: Projects_torch.layers.ConvFeatureExtractionModel,
                 linear_classifier: Projects_torch.layers.LinearClassifier,
                 transformer: Projects_torch.layers.TransformerE
                 ):

        super(SynthTransformerEncoder, self).__init__()

        # self.conv_encoder = conv_encoder

        # self.dropout = nn.Dropout(0.3)

        self.transformer = transformer

        self.linear_classifier = linear_classifier

        self.linear_in = nn.Linear(128, 512)

        self.conv_encoder = conv_encoder

    def forward(self, inputs):

        # outputs = self.conv_encoder(inputs)

        # outputs = self.dropout(outputs)

        inputs = inputs.transpose(dim0=1, dim1=2)

        inputs_ = torch.nn.functional.relu(self.linear_in(inputs))

        encoder_outputs = self.transformer(inputs_)

        return self.linear_classifier(torch.squeeze(encoder_outputs[:, 256:, :], dim=1))
