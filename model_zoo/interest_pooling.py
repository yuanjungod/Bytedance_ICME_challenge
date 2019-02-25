import torch
import torch.nn as nn
import torch.nn.functional as F


class InterestPooling(torch.nn.Module):
    """docstring for InterestPooling"""

    def __init__(self, k, embedding_len, use_cuda=True):
        super(InterestPooling, self).__init__()
        self.k = k
        self.embedding_len = embedding_len
        self.activation = F.sigmoid
        self.d_l1 = nn.Linear(k * embedding_len, 40)
        self.d_l2 = nn.Linear(40, 20)
        self.d_l3 = nn.Linear(20, self.k)
        self.last_activate_layer = nn.Softmax(-1)
        self.use_cuda = use_cuda

    def forward(self, title_embedding_list, bags_Xi):
        title_embedding = torch.cat(title_embedding_list, -1)
        # linear
        din_att = self.d_l1(title_embedding)
        din_att = self.activation(din_att)
        din_att = self.d_l2(din_att)
        din_att = self.activation(din_att)
        din_att = self.d_l3(din_att)

        # Mask for paddings
        paddings = torch.ones(din_att.size()) * (-2 ** 32 + 1)
        if self.use_cuda:
            paddings = paddings.cuda()
        bags_Xi = bags_Xi.unsqueeze(dim=1)
        din_att = torch.where(bags_Xi != 0, din_att, paddings)
        # scale
        # din_att = din_att/(keys.size(-1)**0.5)
        din_att = self.last_activate_layer(din_att)
        # [N,T,1]
        return torch.sum(torch.matmul(din_att, title_embedding), -1)


if __name__ == "__main__":
    pass
