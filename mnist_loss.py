import torch.nn as nn


class Loss(nn.Module):

    def __init__(self):
        super(Loss, self).__init__()

        self.loss = nn.BCELoss()

    def forward(self, outputs, targets):
        return self.loss(outputs, targets)
