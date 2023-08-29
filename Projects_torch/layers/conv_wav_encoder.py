import torch
from torch import nn
from typing import List, Tuple


class ConvFeatureExtractionModel(nn.Module):
    def __init__(self,
                 conv_layers: List[Tuple[int, int, int, int]],
                 num_duplicate_layer: Tuple[int, int, int, int],
                 units: int,
                 is_group_norm: str = True,
                 is_layer_norm: str = False,
                 dropout: float = 0.1,
                 # mode: str = "default",
                 # conv_bias: bool = False
                 ):
        super(ConvFeatureExtractionModel, self).__init__()
        self.conv_layers = None
        self.is_group_norm = is_group_norm
        self.is_layer_norm = is_layer_norm

        def block(layers_param,
                  is_layer_norm,
                  is_group_norm):

            (in_channels, dim, kernel, stride) = layers_param

            def make_conv():
                conv = nn.Conv1d(in_channels=in_channels,
                                 out_channels=dim,
                                 kernel_size=kernel,
                                 stride=stride,
                                 bias=True,
                                 padding=1)
                return conv

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.LayerNorm([512], elementwise_affine=True, eps=1e-6),
                    # nn.Dropout(p=dropout),
                    nn.ReLU(),
                )

            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.GroupNorm(num_groups=32, num_channels=512, eps=1e-6),
                    # nn.Dropout(p=dropout),
                    nn.ReLU(),
                )

            else:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.ReLU(),
                )

        layers = torch.nn.ModuleList()

        for i, layers_param in enumerate(conv_layers):

            for j in range(num_duplicate_layer[i]):
                layers.append(
                    block(
                        layers_param,
                        self.is_layer_norm,
                        self.is_group_norm,
                    )
                )

        self.conv_layers = layers

        self.fc = nn.Linear(in_features=units, out_features=units)
        self.activation = nn.GELU()

    def forward(self, x, **kwargs):
        # BxT -> BxTxC

        for conv in self.conv_layers:
            x = conv(x)
        return x.transpose(dim0=1, dim1=2)


if __name__ == '__main__':
    data = torch.normal(mean=torch.zeros(4, 128, 256))
    conv_layers: List[Tuple[int, int, int, int]] = [(128, 512, 4, 2),
                                                    (512, 512, 3, 1),
                                                    (512, 512, 3, 1),
                                                    (512, 512, 2, 1)]
    num_duplicate_layer: Tuple[int, int, int, int] = (1, 1, 1, 1)
    conv = ConvFeatureExtractionModel(conv_layers=conv_layers,
                                      num_duplicate_layer=num_duplicate_layer,
                                      units=512, )
    output = conv(data)
    print(output.shape)
