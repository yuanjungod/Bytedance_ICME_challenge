from data_io.data_preprocessor import DataPreprocessor
# from model_zoo.deep_fm import DeepFM
from model_zoo.din_xdeep_bert import DeepFM
import torch
import torch.nn as nn
import time
import os
import json
from model_zoo.focal_loss import FocalLoss
import random
from common.logger import logger
from utils.utils import rand_train_data

# 通过下面的方式进行简单配置输出方式与日志级别

# video_db_path = "/Volumes/Seagate Expansion Drive/byte/track2/video.db"
# title_feature_path = "/Volumes/Seagate Expansion Drive/byte/track2/title.db"
# user_db_path = "/Volumes/Seagate Expansion Drive/byte/track2/user.db"
video_db_path = "/Volumes/Seagate Expansion Drive/byte/track2/video.db"
title_feature_path = "/Volumes/Seagate Expansion Drive/byte/track2/title.db"
user_db_path = "/Volumes/Seagate Expansion Drive/byte/track2/user.db"
deep_fm = DeepFM(9, 140000, [80000, 400, 900000, 500, 10, 90000, 80000, 30, 20], 128, 128,
                 embedding_size=64, learning_rate=0.003)
# exit()


"""
    train model
"""
model = deep_fm.train()
# model_path = '/home/yuanjun/code/Bytedance_ICME_challenge/track2/models/finish/byte_115000.model'
# deep_fm.load_state_dict(torch.load(model_path))
model.cuda(0)
# test_dir = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/val_jsons"
# test_dir = "/Volumes/Seagate Expansion Drive/byte/track2/val_jsons"
# test_file_list = [os.path.join(test_dir, i) for i in os.listdir(test_dir)]

# train_dir = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/jsons"
# train_dir = "/Volumes/Seagate Expansion Drive/byte/track2/jsons"

criterion = FocalLoss(2)
# criterion = nn.BCEWithLogitsLoss()
# criterion = nn.BCELoss()
# criterion = F.binary_cross_entropy_with_logits
# F.cross_entropy()
# F.binary_cross_entropy()
# F.cross_entropy()
# torch.nn.BCEloss

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

video_path = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/video.db"
title_path = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/title.db"
interactive_file = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/user.db"
audio_file_path = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/audio.db"
# video_path = "/Volumes/Seagate Expansion Drive/byte/track2/video.db"
# title_path = "/Volumes/Seagate Expansion Drive/byte/track2/title.db"
# interactive_file = "/Volumes/Seagate Expansion Drive/byte/track2/user.db"
# audio_file_path = "/Volumes/Seagate Expansion Drive/byte/track2/audio.db"
data_prepro_tool = DataPreprocessor(video_path, interactive_file, title_path, audio_file_path)
result_list = data_prepro_tool.get_train_data()
for epoch in range(total_epochs):

    # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$epoch: %s$$$$$$$$$$$$$$$$$$$$$$$$$$$" % epoch)
    logger.info("$$$$$$$$$$$$$$$$$$$$$$$$$$$epoch: %s$$$$$$$$$$$$$$$$$$$$$$$$$$$" % epoch)
    # for result in os.listdir(train_dir):
    #     fp = open(os.path.join(train_dir, result), "r")
    #     result = json.load(fp)
    #     fp.close()
    #
    #     result = rand_train_data(result)
    #
    #     val_fp = open(random.choice(test_file_list), "r")
    #     val_result = json.load(val_fp)
    #     val_fp.close()
    for result in result_list:

        # print("%s data load finished" % count, time.time() - load_data_time)
        # print(result["index"][0])
        deep_fm.fit2(model, optimizer, criterion, result["index"], result["value"], result["video"], result['audio'],
                     result["title"], result["title_value"], result["like"], result["finish"], count,
                     save_path="/Volumes/Seagate Expansion Drive/byte/track2/models/",
                     Xi_valid=None, Xv_valid=None, y_like_valid=None, y_finish_valid=None,
                     video_feature_val=None, title_feature_val=None, title_value_val=None, total_epochs=total_epochs)
        count += 1
        load_data_time = time.time()

