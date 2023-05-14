
import torch
import Projects_torch.layers
from typing import List, Tuple


class SynthAutoEncoder(torch.nn.Module):

    transformer: Projects_torch.layers.Transformer
    linear_classifier: Projects_torch.layers.LinearClassifier

    def __init__(self,
                 conv_encoder: Projects_torch.layers.ConvFeatureExtractionModel,
                 conv_decoder: Projects_torch.layers.ConvDecoderModel,
                 transformer: Projects_torch.layers.Transformer,
                 linear_classifier: Projects_torch.layers.LinearClassifier,
                 ):

        super(SynthAutoEncoder, self).__init__()

        self.transformer = transformer

        self.linear_classifier = linear_classifier

        self.conv_encoder = conv_encoder
        self.conv_decoder = conv_decoder

    def forward(self, inputs):

        inputs1, inputs2 = inputs

        outputs = self.conv_encoder(inputs1) # output:(batch, t, channel)

        decoder_outputs, encoder_outputs = self.transformer([outputs, inputs2])

        outputs_params_list = self.linear_classifier(encoder_outputs)

        stft_outputs = self.conv_decoder(decoder_outputs)

        stft = torch.cat([inputs1, stft_outputs], dim=0)

        return outputs_params_list, stft


if __name__ == '__main__':
    data = torch.random(size=(2, 32, 32))
    conv_layers: List[Tuple[int, int, int]] = [(512, 10, 5),
                                               (512, 3, 2),
                                               (512, 3, 2),
                                               (512, 3, 2),
                                               (512, 3, 2),
                                               (512, 2, 2),
                                               (512, 2, 2)]

    num_duplicate_layer: Tuple[int, int, int, int, int, int, int] = (1, 1, 1, 1, 1, 1, 1)
    conv = layers.ConvFeatureExtractionModel(conv_layers=conv_layers, activation='gelu', units=512,
                                             num_duplicate_layer=num_duplicate_layer)

    inputs = tf.random.normal((2, 16384, 1))

    encoder = layers.EncoderTransformer(num_layers=8, d_model=512, num_attention_heads=8, dff=768, dropout_rate=0.1,
                                        activation='relu')

    inputs_shape: Tuple[int, int] = (16384, 1)

    model = SynthEncoder(conv_encoder=conv,
                         transformer_encoder=encoder,
                         inputs=inputs_shape,
                         top_k_transformer=2)

    m = model.build()
    # outputs = m(inputs)
    # print(outputs.shape)
    m.summary()
