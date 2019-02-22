from data_io.data_preprocessor import DataPreprocessor
from model_zoo.deep_fm import DeepFM
import torch
import torch.nn.functional as F
import time
import os
import json


# video_db_path = "/Volumes/Seagate Expansion Drive/byte/track2/video.db"
# title_feature_path = "/Volumes/Seagate Expansion Drive/byte/track2/title.db"
# user_db_path = "/Volumes/Seagate Expansion Drive/byte/track2/user.db"
video_db_path = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/video.db"
title_feature_path = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/title.db"
user_db_path = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/user.db"
deep_fm = DeepFM(9, 140000, [80000, 400, 900000, 500, 10, 90000, 80000, 30, 20], 128, "like")

"""
    train model
"""
model = deep_fm.train()
model.cuda(0)

optimizer = torch.optim.SGD(model.parameters(), lr=model.learning_rate, weight_decay=model.weight_decay)
if model.optimizer_type == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate, weight_decay=model.weight_decay)
elif model.optimizer_type == 'rmsp':
    optimizer = torch.optim.RMSprop(model.parameters(), lr=model.learning_rate, weight_decay=model.weight_decay)
elif model.optimizer_type == 'adag':
    optimizer = torch.optim.Adagrad(model.parameters(), lr=model.learning_rate, weight_decay=model.weight_decay)

criterion = F.binary_cross_entropy_with_logits

count = 0
load_data_time = time.time()
for _ in range(5):
    for result in os.listdir("/home/yuanjun/code/Bytedance_ICME_challenge/track2/jsons"):
        fp = open(os.path.join("/home/yuanjun/code/Bytedance_ICME_challenge/track2/jsons", result), "r")
        result = json.load(fp)
        fp.close()
        # print("%s data load finished" % count, time.time() - load_data_time)
        # print(result["index"][0])
        deep_fm.fit2(model, optimizer, criterion, result["index"], result["value"], result["video"], result["title"],
                     result["title_value"], result["like"], result["finish"], count,
                     save_path="/Users/quantum/code/Bytedance_ICME_challenge/")
        count += 1
        load_data_time = time.time()

