from data_io.data_preprocessor import DataPreprocessor
from model_zoo.deep_fm import DeepFM
import torch
import torch.nn as nn
import time
import os
import json
from model_zoo.focal_loss import FocalLoss
import random
import logging
from utils.utils import rand_train_data
from data_io.data_preprocessor import DataPreprocessor

# 通过下面的方式进行简单配置输出方式与日志级别

video_db_path = "/Volumes/Seagate Expansion Drive/byte/track2/video.db"
title_feature_path = "/Volumes/Seagate Expansion Drive/byte/track2/title.db"
user_db_path = "/Volumes/Seagate Expansion Drive/byte/track2/user.db"
# video_db_path = "/Volumes/Seagate Expansion Drive/byte/track2/video.db"
# title_feature_path = "/Volumes/Seagate Expansion Drive/byte/track2/title.db"
# user_db_path = "/Volumes/Seagate Expansion Drive/byte/track2/user.db"
task = "finish"
deep_fm = DeepFM(9, 140000, [80000, 400, 900000, 500, 10, 90000, 80000, 30, 20], 128, 128,
                 embedding_size=64, learning_rate=0.003)
# exit()
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', filename='%s_logger.log' % task, level=logging.INFO)

"""
    train model
"""
model = deep_fm.train()
# model_path = '/home/yuanjun/code/Bytedance_ICME_challenge/track2/models/finish/byte_115000.model'
# deep_fm.load_state_dict(torch.load(model_path))
model.cuda(0)
test_dir = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/val_jsons"
# test_dir = "/Volumes/Seagate Expansion Drive/byte/track2/val_jsons"
test_file_list = [os.path.join(test_dir, i) for i in os.listdir(test_dir)]

train_dir = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/jsons"
# train_dir = "/Volumes/Seagate Expansion Drive/byte/track2/jsons"

criterion = FocalLoss(2)
# criterion = nn.BCELoss()
# criterion = F.binary_cross_entropy_with_logits
# F.cross_entropy()
# F.binary_cross_entropy()
# F.cross_entropy()
# torch.nn.BCEloss
video_path = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/track2_video_features.txt"
title_path = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/track2_title.txt"
interactive_file = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/final_track2_train.txt"
audio_file_path = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/track2_audio_features.txt"
data_prepro_tool = DataPreprocessor()
result, val_result = data_prepro_tool.get_train_data_from_origin_file(
    video_path, title_path, interactive_file, audio_file_path)

count = 0
load_data_time = time.time()

optimizer = torch.optim.SGD(model.parameters(), lr=model.learning_rate, weight_decay=model.weight_decay)
if model.optimizer_type == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate, weight_decay=model.weight_decay)
elif model.optimizer_type == 'rmsp':
    optimizer = torch.optim.RMSprop(model.parameters(), lr=model.learning_rate, weight_decay=model.weight_decay)
elif model.optimizer_type == 'adag':
    optimizer = torch.optim.Adagrad(model.parameters(), lr=model.learning_rate, weight_decay=model.weight_decay)

total_epochs = 3

for epoch in range(total_epochs):

    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$epoch: %s$$$$$$$$$$$$$$$$$$$$$$$$$$$" % epoch)
    logging.info("$$$$$$$$$$$$$$$$$$$$$$$$$$$epoch: %s$$$$$$$$$$$$$$$$$$$$$$$$$$$" % epoch)

    # print("%s data load finished" % count, time.time() - load_data_time)
    # print(result["index"][0])
    deep_fm.fit2(model, optimizer, criterion, result["index"], result["value"], result["video"], result["audio"],
                 result["title"], result["title_value"], result["like"], result["finish"], count,
                 save_path="/home/yuanjun/code/Bytedance_ICME_challenge/track2/models/%s" % task,
                 Xi_valid=val_result["index"], Xv_valid=val_result["value"],
                 y_like_valid=val_result["like"], y_finish_valid=val_result["finish"],
                 video_feature_val=val_result["video"], title_feature_val=val_result["title"],
                 title_value_val=val_result["title_value"], total_epochs=total_epochs)
    count += 1
    load_data_time = time.time()

