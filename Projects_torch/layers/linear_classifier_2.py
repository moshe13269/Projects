import torch
from torch import nn
from typing import List, Tuple


class LinearClassifier(nn.Module):
    """
    Transformer Encdoer outputs are: (batch, t', channels)
    Given N classes (types of parameters) when all specific class contain num_classes
    The Dense layer mapping latent space to accurate outputs:
    outputs -> class_i: (,t', channels) -> (, num_classes) and softmax

    """

    outputs_dimension_per_outputs: List[int]
    activation: str = 'softmax'
    dropout: float = 0.2

    def __init__(self,
                 outputs_dimension_per_outputs,
                 activation: str = 'relu',
                 dropout: float = 0.2,
                 **kwargs):
        super().__init__(**kwargs)

        self.outputs_dimension_per_outputs = outputs_dimension_per_outputs
        self.activation = activation
        self.layers = torch.nn.ModuleList()
        for output_size in self.outputs_dimension_per_outputs:
            self.layers.append(
                nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(in_features=65 * 512, out_features=output_size),
                    nn.Dropout(p=dropout),
                    nn.ReLU(),
                    nn.Linear(in_features=output_size,
                              out_features=output_size),
                    nn.Softmax(dim=-1)
                )
            )

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            outputs.append(layer(x))
        return outputs
        #torch.nn.ModuleList() ???


if __name__ == "__main__":
    linear = LinearClassifier([3, 12, 20, 31, 4, 5, 8, 5, 16])
    input = torch.normal(torch.zeros(32, 130, 512))
    output = linear(input)
    loss = nn.CrossEntropyLoss()
    target = torch.empty((32,), dtype=torch.long).random_(3)
    output = loss(output[0], target)
    print(output)
