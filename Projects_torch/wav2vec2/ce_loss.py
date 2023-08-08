
import torch
from torch import nn


class CELoss(nn.Module):
    """
    inputs: (batch, c') where c'=sum(outputs_dimension_per_outputs)

    """

    def __init__(self, outputs_dimension_per_outputs):

        super(CELoss, self).__init__()

        self.outputs_dimension_per_outputs = outputs_dimension_per_outputs
        self.index2split = [sum(self.outputs_dimension_per_outputs[:i])
                            for i in range(len(self.outputs_dimension_per_outputs) + 1)]

        self.ce = nn.CrossEntropyLoss()

    def forward(self, output, target):

        loss = 0.0

        output_list = [output[:, self.index2split[i]: self.index2split[i+1]] for i in range(len(self.index2split)-1)]

        for i in range(len(output_list)):
            loss += self.ce(torch.nn.functional.softmax(output_list[i], dim=-1), target[:, i:i+1].squeeze())
        loss = loss / len(output_list)

        return loss
