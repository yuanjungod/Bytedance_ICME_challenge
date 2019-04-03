import torch
from model_zoo.sparse_adam import SparseAdam
import random
# optimizer = torch.optim.Adam()


class SparseModel(torch.nn.Module):
    def __init__(self):
        super(SparseModel, self).__init__()
        self.embedding1 = torch.nn.Embedding(100000, 6, sparse=True)
        self.liner1 = torch.nn.Linear(6, 1)

    def forward(self, input):
        b = self.embedding1(a)
        b = self.liner1(b)
        return b


sparse_params = list()
dense_params = list()
test_model = SparseModel()
for name, value in test_model.named_parameters():
    # print(type(value))
    print(name)
    if name.find("embedding") != -1:
        sparse_params.append({
            "params": value,
            # "lr": 1e-3
        })
    else:
        dense_params.append({
            "params": value,
            # "lr": 1e-3
        })

# optimizer = torch.optim.Adam(test_model.liner1.parameters())
optimizer = torch.optim.Adam(dense_params)
# sparse_optimizer = SparseAdam(test_model.embedding1.parameters())
sparse_optimizer = SparseAdam(sparse_params, weight_decay=0.001)
for i in range(100):
    a = torch.LongTensor([random.choice([i for i in range(10000)]) for _ in range(6)])
    # print("input", a)
    b = test_model(a)
    criterion = torch.nn.BCEWithLogitsLoss(weight=torch.FloatTensor([10]))
    b = b.view(-1)
    loss = criterion(b, torch.FloatTensor([0, 1, 1, 0, 1, 0]))
    optimizer.zero_grad()
    sparse_optimizer.zero_grad()
    loss.backward()
    print(loss)
    optimizer.step()
    sparse_optimizer.step()



