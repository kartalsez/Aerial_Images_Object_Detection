# Source: https://github.com/ycszen/pytorch-segmentation/blob/master/loss.py

import torch
import torch.nn.functional as F


class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None, reduction='elementwise_mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = torch.nn.NLLLoss(weight, reduction=reduction)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)
