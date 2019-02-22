import torch
import torch.nn as nn

value_list = [[1, 1, 1], [1, 1, 0]]
a = torch.LongTensor([[1, 2, 3], [13, 14, 0]])
b = torch.FloatTensor([[[j for _ in range(5)] for j in i] for i in value_list])

embedding = nn.Embedding(100, 5)
embedding_a = embedding(a)
size = embedding_a.shape
# embedding_a = embedding_a.view(size[0], size[2], size[1])
print(embedding_a)
print(embedding_a.shape)
print(b.shape)
# mat_mul = torch.matmul(embedding_a, b)
# mat_mul = embedding_a * b
mat_mul = embedding_a*b
print(mat_mul)
mat_mul = mat_mul.permute(0, 2, 1)
mat_mul = torch.sum(mat_mul, 1)
print(mat_mul)
print(mat_mul.shape)

c = torch.LongTensor([[[1], [2], [3]], [[13], [14], [0]]])
d = torch.LongTensor([value_list])
print("#"*20)
print(c)
print(embedding(c[1, :, :]))
print(c.shape)
# fm_first_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in
#                                       enumerate(self.fm_first_order_embeddings)]