
import torch
from torch import nn


class CELoss(nn.Module):
    """
    inputs: (batch, c') where c'=sum(outputs_dimension_per_outputs)

    """

    def __init__(self, outputs_dimension_per_outputs,
                 num_classes,
                 index_y_true,
                 autoencoder=False):
        super(CELoss, self).__init__()
        self.num_classes = num_classes
        self.index_y_true = index_y_true

        self.outputs_dimension_per_outputs = outputs_dimension_per_outputs
        self.index2split = [sum(self.outputs_dimension_per_outputs[:i])
                            for i in range(len(self.outputs_dimension_per_outputs) + 1)]

        self.ce = nn.CrossEntropyLoss()
        self.autoencoder = autoencoder

    def forward(self, output, target):

        loss = 0.0

        if self.autoencoder:
            for output_ in output:
                if len(output_) > 2:

                    for i in range(len(target)):
                        loss += self.ce(output_[i], target[1][i].squeeze())
                    loss = loss / len(output_)
        else:
            for i in range(len(target)):
                loss += self.ce(output[i], target[i].squeeze())
            loss = loss / len(target)

        return loss
