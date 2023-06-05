import torch
import torch.nn as nn
import Projects_torch.layers
from typing import List, Tuple


class SynthTransformerAutoEncoder(nn.Module):
    num_layers: int
    d_model: int
    nhead: int

    linear_classifier: Projects_torch.layers.LinearClassifier

    def __init__(self,
                 conv_encoder: Projects_torch.layers.ConvFeatureExtractionModel,
                 linear_classifier: Projects_torch.layers.LinearClassifier,
                 transformer: Projects_torch.layers.Transformer
                 ):

        super(SynthTransformerAutoEncoder, self).__init__()

        self.transformer = transformer

        self.linear_classifier = linear_classifier

        self.conv_encoder = conv_encoder

    def forward(self, inputs):

        outputs = self.conv_encoder(inputs)

        output, encoder_outputs = self.transformer(outputs)

        outputs_params_list = self.linear_classifier(encoder_outputs)

        return output, outputs_params_list