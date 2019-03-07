import torch
import torch.nn as nn
import random
import time

a = [[random.random() for _ in range(10)] for _ in range(500000)]
b = [random.random() for _ in range(10)]
# b = a[0]
c = list()
start = time.time()
# random.shuffle(a)
for i in a:
    c.append(i)
print("consume: ", time.time() - start)
