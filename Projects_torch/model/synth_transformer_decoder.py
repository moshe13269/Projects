import torch.nn as nn
from Projects_torch.layers.transformer_decoder import TransformerD


class SynthTransformerDecoder(nn.Module):

    def __init__(self,
                 transformer: TransformerD #Projects_torch.layers.transformer_decoder.TransformerD
                 ):

        super(SynthTransformerDecoder, self).__init__()

        self.transformer = transformer

    def forward(self, inputs):
        decoder_inputs, condition_vector = inputs

        outputs = self.transformer(decoder_inputs=decoder_inputs, condition_vector=condition_vector)

        return outputs
