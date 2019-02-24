from data_io.data_preprocessor import DataPreprocessor
from model_zoo.deep_fm import DeepFM
import torch
import torch.nn.functional as F
import time
import os
import json
from model_zoo.focal_loss import FocalLoss
import random
import logging

# 通过下面的方式进行简单配置输出方式与日志级别

video_db_path = "/Volumes/Seagate Expansion Drive/byte/track2/video.db"
title_feature_path = "/Volumes/Seagate Expansion Drive/byte/track2/title.db"
user_db_path = "/Volumes/Seagate Expansion Drive/byte/track2/user.db"
# video_db_path = "/Volumes/Seagate Expansion Drive/byte/track2/video.db"
# title_feature_path = "/Volumes/Seagate Expansion Drive/byte/track2/title.db"
# user_db_path = "/Volumes/Seagate Expansion Drive/byte/track2/user.db"
task = "finish"
deep_fm = DeepFM(9, 140000, [80000, 400, 900000, 500, 10, 90000, 80000, 30, 20], 128, task,
                 embedding_size=64, learning_rate=0.003, weight_decay=0.00005)

logging.basicConfig(filename='%s_logger.log' % task, level=logging.INFO)

"""
    train model
"""
model = deep_fm.train()
# model_path = '/home/yuanjun/code/Bytedance_ICME_challenge/track2/models/like/20190224/byte_90000.model'
# deep_fm.load_state_dict(torch.load(model_path))
model.cuda(0)
test_dir = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/val_jsons"
test_file_list = [os.path.join(test_dir, i) for i in os.listdir(test_dir)]


# criterion = FocalLoss(2)
criterion = F.binary_cross_entropy_with_logits

count = 0
load_data_time = time.time()
for epoch in range(3):
    optimizer = torch.optim.SGD(model.parameters(), lr=model.learning_rate/(10**epoch), weight_decay=model.weight_decay)
    if model.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate/(10**epoch), weight_decay=model.weight_decay)
    elif model.optimizer_type == 'rmsp':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=model.learning_rate/(10**epoch), weight_decay=model.weight_decay)
    elif model.optimizer_type == 'adag':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=model.learning_rate/(10**epoch), weight_decay=model.weight_decay)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$epoch: %s$$$$$$$$$$$$$$$$$$$$$$$$$$$" % epoch)
    logging.info("$$$$$$$$$$$$$$$$$$$$$$$$$$$epoch: %s$$$$$$$$$$$$$$$$$$$$$$$$$$$" % epoch)
    for result in os.listdir("/home/yuanjun/code/Bytedance_ICME_challenge/track2/jsons"):
        fp = open(os.path.join("/home/yuanjun/code/Bytedance_ICME_challenge/track2/jsons", result), "r")
        result = json.load(fp)
        fp.close()

        val_fp = open(random.choice(test_file_list), "r")
        val_result = json.load(val_fp)
        val_fp.close()

        # print("%s data load finished" % count, time.time() - load_data_time)
        # print(result["index"][0])
        deep_fm.fit2(model, optimizer, criterion, result["index"], result["value"], result["video"], result["title"],
                     result["title_value"], result["like"], result["finish"], count,
                     save_path="/home/yuanjun/code/Bytedance_ICME_challenge/track2/models/%s" % task,
                     Xi_valid=val_result["index"], Xv_valid=val_result["value"],
                     y_like_valid=val_result["like"], y_finish_valid=val_result["finish"],
                     video_feature_val=val_result["video"], title_feature_val=val_result["title"],
                     title_value_val=val_result["title_value"])
        count += 1
        load_data_time = time.time()

