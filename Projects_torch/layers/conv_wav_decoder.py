from torch import nn
from typing import List, Tuple


class ConvDecoderModel(nn.Module):
    def __init__(self,
                 conv_layers: List[Tuple[int, int, int]],
                 num_duplicate_layer: Tuple[int, int, int, int],
                 activation: str,
                 units: int,
                 is_group_norm: str = True,
                 is_layer_norm: str = False,
                 dropout: float = 0.0,
                 mode: str = "default",
                 conv_bias: bool = False):
        super(ConvDecoderModel, self).__init__()
        self.conv_layers = None

        def block(layers_param,
                  activation,
                  is_layer_norm=False,
                  is_group_norm=True,
                  conv_bias=False):

            (in_channels, dim, kernel, stride) = layers_param  # (dim, kernel, stride, output_padding) = layers_param

            def make_conv():
                conv = nn.ConvTranspose1d(in_channels=in_channels,
                                          out_channels=dim,
                                          kernel_size=kernel,
                                          stride=stride,
                                          bias=conv_bias,
                                          padding=1,
                                          )
                return conv

            assert (is_layer_norm and is_group_norm) == False, "layer norm and group norm are exclusive"

            if is_layer_norm and activation is not None:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.LayerNorm([512], elementwise_affine=True, eps=1e-6),
                    nn.GELU(),
                )

            elif is_group_norm and activation is not None:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.GroupNorm(num_groups=32, num_channels=512, eps=1e-6),
                    nn.GELU(),
                )

            else:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.GELU(),
                )

        layers = []

        for i, layers_param in enumerate(conv_layers):

            for j in range(num_duplicate_layer[i]):
                layers.append(
                    block(
                        layers_param,
                        activation,
                        is_layer_norm=mode == "layer_norm",
                        is_group_norm=mode == "default",
                        conv_bias=conv_bias,
                    )
                )

        self.conv_layers = layers

        # self.avg_pool = nn.AvgPool1d()

        self.fc = nn.Linear(in_features=units, out_features=1)

        self.activation = nn.GELU()

    def forward(self, x, **kwargs):
        # BxT -> BxTxC
        for conv in self.conv_layers:
            x = conv(x)
        return self.activation(self.fc(x))


if __name__ == '__main__':
    # conv_layers: List[Tuple[int, int, int, int]] = [(512, 2, 2, 1),
    #                                                 (512, 2, 2, 1),
    #                                                 (512, 3, 2, 0),
    #                                                 (512, 3, 2, 0),
    #                                                 (512, 3, 2, 0),
    #                                                 (512, 3, 2, 0),
    #                                                 (512, 4, 2, 0)]

    # [(512, 2, 2, 1),
    #  (512, 2, 2, 1),
    #  (512, 3, 2, 1),
    #  (512, 3, 2, 1),
    #  (512, 3, 2, 0),
    #  (512, 3, 2, 0),
    #  (512, 10, 5, 4)]
    # num_duplicate_layer: Tuple[int, int, int, int, int, int, int] = (1, 1, 1, 1, 1, 1, 1)
    conv_layers: List[Tuple[int, int, int]] = [(512, 2, 1),
                                               (512, 3, 1),
                                               (512, 3, 1),
                                               (512, 4, 2)]
    num_duplicate_layer: Tuple[int, int, int, int] = (1, 1, 1, 1)
    conv = ConvDecoderModel(conv_layers=conv_layers, activation='gelu', units=129,
                            num_duplicate_layer=num_duplicate_layer)
    # output = conv(data)
    # print(output.shape)
    # print(tf.reduce_mean(output), tf.reduce_min(output), tf.reduce_max(output))
