import torch
from torch import nn


class L2Loss(nn.Module):

    def __init__(self):
        super(L2Loss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, output, target):
        # student, teacher = torch.split(output, split_size_or_sections=output.shape[0]//2, dim=0)
        loss = 0.0
        for i in range(len(output)):
            if len(output[i]) == 2:
                self.loss(output[i][0], output[i][1])
        return loss
