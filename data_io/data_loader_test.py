from torch.utils.data import Dataset, DataLoader
import time
import json
from tqdm import tqdm, trange
# from common.logger import logger
from multiprocessing.managers import BaseManager
import multiprocessing as mp
import random
import os
import math
from pymemcache.client.base import Client

host = "localhost"

client = Client((host, 11211))

start = time.time()
# for i in range(1000):
#     for i in range(1000):
#         client.get("%s" % i)
# print("memcache", time.time()-start)
# exit()

# result = client.get('some_key')

user_list = mp.Manager().list()
title = mp.Manager().dict()
title_value = mp.Manager().dict()
video_dict = mp.Manager().dict()
audio_dict = mp.Manager().dict()


# normal_dict = dict()
#
length = 10000
#
# for i in range(length):
#     if i % 1000 == 0:
#         print(i)
#     user_list.append(i)
#     title[i] = json.dumps([random.random() for _ in range(512)])
#     title_value[i] = json.dumps([random.random() for _ in range(512)])
#     video_dict[i] = json.dumps([random.random() for _ in range(512)])
#     audio_dict[i] = json.dumps([random.random() for _ in range(512)])
#
#     normal_dict[i] = json.dumps([random.random() for _ in range(512)])

    # client.set("%s" % i, json.dumps([random.random() for _ in range(512)]))
test = [0 for _ in range(100)]


class trainset(Dataset):

    def __init__(self):
        self.client = None

    def __getitem__(self, index):
        if self.client is None:
            self.client = Client((host, 11211))
        # time.sleep(1)
        # print(os.getpid())
        # print(id(user_list))
        # time.sleep(0.01)

        # user = user_list[index]

        # start = time.time()
        # a = json.loads(title[user])
        # for i in range(10000):
        #     a = 999
        #     math.sqrt(a)
        # print(time.time()-start)
        # return test
        # print(self.client.get("%s" % random.choice(range(1000))))
        return [json.loads(self.client.get("%s" % (index % 1000))) for _ in range(4)]

    def __len__(self):
        return length


start = time.time()
# manager = Manager2()
# data = manager.Data()
# a = manager.Trainset()
a = trainset()
for epoch in trange(int(1), desc="Epoch"):
    # print(epoch)
    for i in DataLoader(a, batch_size=10000, shuffle=True, num_workers=1):
        # print(i)
        # print(time.time() - start)
        # time.sleep(13)
        # start = time.time()
        # logger.info("ok")
        # start = time.time()
        pass
print(time.time() - start)

# start = time.time()
# for i in range(10):
#     for i in range(1000):
#         a = normal_dict[i]
# print("normal_dict", time.time()-start)
#
# start = time.time()
# for i in range(10):
#     for i in range(1000):
#         a = audio_dict[i]
# print("share_mem", time.time()-start)
