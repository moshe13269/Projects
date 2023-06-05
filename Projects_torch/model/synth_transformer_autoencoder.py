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
                 conv_decoder: Projects_torch.layers.conv_wav_decoder.ConvDecoderModel,
                 linear_classifier: Projects_torch.layers.LinearClassifier,
                 transformer: Projects_torch.layers.Transformer
                 ):

        super(SynthTransformerAutoEncoder, self).__init__()

        self.transformer = transformer

        self.linear_classifier = linear_classifier

        self.conv_encoder = conv_encoder
        self.conv_decoder = conv_decoder

    def forward(self, inputs):
        outputs = self.conv_encoder(inputs)

        output, encoder_outputs = self.transformer(src=outputs, tgt=outputs)

        outputs_params_list = self.linear_classifier(encoder_outputs)

        stft = self.conv_decoder(output)

        return [inputs, stft], [outputs, output], outputs_params_list
