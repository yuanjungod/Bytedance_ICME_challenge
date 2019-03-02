import torch
import torch.nn as nn


class InterestPooling1(torch.nn.Module):

    def __init__(self, k, embedding_len, use_cuda=True):
        super(InterestPooling1, self).__init__()
        self.k = k
        self.embedding_len = embedding_len
        self.activation = torch.sigmoid
        self.d_l1 = nn.Linear(k * embedding_len, 40)
        self.d_l2 = nn.Linear(40, 20)
        self.d_l3 = nn.Linear(20, self.k)
        self.last_activate_layer = nn.Softmax(-1)
        self.use_cuda = use_cuda

    def forward(self, title_embedding, bags_Xi):
        # title_embedding = torch.cat(title_embedding_list, -1)
        # linear
        # title_embedding.view(-1, self.k*self.embedding_len)
        din_att = self.d_l1(title_embedding)
        din_att = self.activation(din_att)
        din_att = self.d_l2(din_att)
        din_att = self.activation(din_att)
        din_att = self.d_l3(din_att)
        # print("din_att1", din_att)
        # Mask for paddings
        paddings = torch.ones(din_att.size()) * (-2 ** 32 + 1)
        if self.use_cuda:
            print(type(paddings))
            # paddings = paddings.cuda()
        # bags_Xi = bags_Xi.unsqueeze(dim=1)
        # print("bags_Xi", bags_Xi)
        din_att = torch.where(bags_Xi != 0, din_att, paddings)
        # print("din_att2", din_att)
        # scale
        # din_att = din_att/(keys.size(-1)**0.5)
        din_att = self.last_activate_layer(din_att)
        # print("din_att3", din_att)
        # [N,T,1]
        title_embedding = title_embedding.view(-1, self.k, self.embedding_len)
        # print("title_embedding", title_embedding.size())
        return torch.einsum('ij,ijk->ik', [din_att, title_embedding])
        # return torch.matmul(din_att, title_embedding)


class InterestPooling2(torch.nn.Module):
    """docstring for InterestPooling"""

    def __init__(self, k, use_cuda=True):
        super(InterestPooling2, self).__init__()
        self.k = k
        self.activation = torch.sigmoid
        self.d_l1 = nn.Linear(k * 4, 40)
        self.d_l2 = nn.Linear(40, 20)
        self.d_l3 = nn.Linear(20, 1)
        self.last_activate_layer = nn.Softmax(-1)
        self.use_cuda = use_cuda

    def forward(self, queries, keys, bags_Xi):
        """
        Args:
            queries: [N,1,K]
            keys: [N,T,K]
            bags_Xi:[N,T]
        """
        queries = queries.repeat(1, keys.size(1), 1)  # [N,T,K]
        din_all = torch.cat([queries, keys, queries - keys, queries * keys], -1)  # [N,T,4K]
        # linear
        din_att = self.d_l1(din_all)
        din_att = self.activation(din_att)
        din_att = self.d_l2(din_att)
        din_att = self.activation(din_att)
        din_att = self.d_l3(din_att)  # [N,T,1]

        din_att = din_att.view(-1, 1, keys.size(1))  # [N,1,T]

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
        din_att = din_att.view(-1, keys.size(1))
        # return torch.matmul(din_att, keys)  # [N,1,K]
        return torch.einsum('ij,ijk->ik', [din_att, keys])


if __name__ == "__main__":
    import random
    # interest_pool = InterestPooling1(4, 30, False)
    # title_embedding = torch.FloatTensor([[random.random() for _ in range(120)]])
    # bags_xi = torch.FloatTensor([[1, 0, 1, 0]])
    # print(interest_pool(title_embedding, bags_xi))
    interest_pool = InterestPooling2(30, False)
    queries = torch.FloatTensor([[[random.random() for _ in range(30)]]])
    keys = torch.FloatTensor([[[random.random() for _ in range(30)] for i in range(30)]])
    bags = torch.LongTensor([[random.choice([0, 1]) for _ in range(30)]])
    print(interest_pool(queries, keys, bags).size())
