import torch
from torch import nn
import torch.nn.functional as F


class Dice(nn.Module):

    def __init__(self, input_size, axis=-1, epsilon=1e-9):
        super(Dice, self).__init__()
        self.axis = axis
        self.epsilon = epsilon
        self.bn = nn.BatchNorm1d(input_size)
        self.alphas = torch.nn.Parameter(torch.ones(input_size))
        self.reset_parameters()

    def reset_parameters(self):
        self.alphas.data.uniform_()

    def forward(self, inputs):
        inputs_normed = self.bn(inputs)
        x_p = torch.sigmoid(inputs_normed)
        return self.alphas * (1.0 - x_p) * inputs + x_p * inputs


class ParametricRelu(nn.Module):

    def __init__(self, input_size):
        super(ParametricRelu, self).__init__()
        self.alphas = torch.nn.Parameter(torch.ones(input_size))
        self.reset_parameters()

    def reset_parameters(self):
        self.alphas.data.uniform_()

    def forward(self, inputs):
        pos = F.relu(inputs)
        neg = self.alphas * (inputs - abs(inputs)) * 0.5
        return pos+neg


if __name__ == "__main__":
    import random
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm, trange


    class Deep(torch.nn.Module):
        def __init__(self, input_size):
            super(Deep, self).__init__()
            self.line1 = nn.Linear(input_size, 50)
            self.activate1 = Dice(50)
            self.line2 = nn.Linear(50, 20)
            self.activate2 = Dice(20)
            self.line3 = nn.Linear(20, 1)

        def forward(self, input):
            a = self.line1(input)
            a = self.activate1(a)
            a = self.line2(a)
            a = self.activate2(a)
            return self.line3(a).reshape(-1)

    model = Deep(100)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.1)

    train_x = torch.FloatTensor([[random.random() for _ in range(100)] for _ in range(200)])
    train_y = torch.FloatTensor([1 if sum(i) > len(i)/2 else 0 for i in train_x])
    data_set = TensorDataset(train_x, train_y)
    train_dataloader = DataLoader(data_set, batch_size=10, shuffle=True)
    for epoch in trange(1000, desc="Epoch"):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch_train_x, batch_train_y = batch
            output = model(batch_train_x)
            # print(output)
            # exit()
            loss = criterion(output, batch_train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if random.random() > 0.993:
                print(loss)

