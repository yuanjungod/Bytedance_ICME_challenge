import torch.nn as nn
import torch.nn.functional as F
import torch


# https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/utils/misc.py
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss((1 - F.softmax(inputs, dim=-1)) ** self.gamma * F.log_softmax(inputs, dim=-1), targets)


def bce_focal_loss(input, target, gamma=2, weight=None, size_average=True, reduce=True):
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

    prob = torch.sigmoid(input)
    prob_for_gt = target * prob + (1 - target) * (1 - prob)

    if weight is not None:
        loss = loss * weight

    loss = loss * torch.pow((1 - prob_for_gt), gamma)
    # print(torch.pow((1-prob_for_gt),gamma))

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()

