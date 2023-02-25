import torch
from torch import nn


class L2Loss(nn.Module):

    def __init__(self):
        super(L2Loss, self).__init__()
        self.loss = nn.MSELoss()

    def forword(self, output, target):
        student, teacher = torch.split(output, split_size_or_sections=2, dim=0)
        return self.loss(student, teacher)
