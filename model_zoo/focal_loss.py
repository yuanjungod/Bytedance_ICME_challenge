import torch.nn as nn
import torch.nn.functional as F
import torch


# https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/utils/misc.py
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.bce_loss = nn.BCELoss(weight=weight, size_average=size_average)

    def forward(self, inputs, targets):
        print(-(1 - F.sigmoid(inputs)) ** self.gamma * F.logsigmoid(inputs))
        return self.bce_loss(-(1 - F.sigmoid(inputs)) ** self.gamma * F.logsigmoid(inputs), targets)


if __name__ == "__main__":
    input = torch.FloatTensor([1, 2])
    target = torch.FloatTensor([1, 0])
    focal_loss = FocalLoss()
    loss = focal_loss(input, target)
    print(loss)

