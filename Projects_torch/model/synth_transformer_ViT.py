import torch.nn as nn
import Projects_torch.layers


class SynthTransformerViT(nn.Module):
    num_layers: int
    d_model: int
    nhead: int

    linear_classifier: Projects_torch.layers.LinearClassifier

    def __init__(self,
                 # conv_encoder: Projects_torch.layers.ConvFeatureExtractionModel,
                 # linear_classifier: Projects_torch.layers.LinearClassifier,
                 transformer: Projects_torch.layers.TransformerE
                 ):

        super(SynthTransformerViT, self).__init__()

        self.transformer = transformer

    def forward(self, inputs):

        encoder_outputs = self.transformer(inputs)

        return encoder_outputs
