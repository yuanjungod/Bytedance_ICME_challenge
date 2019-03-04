from data_io.data_preprocessor import DataPreprocessor
from model_zoo.din_xdeep_bert import DeepFM
import torch
import torch.nn as nn
import time
import os
import json
from model_zoo.focal_loss import FocalLoss
import random
import logging
from utils.utils import rand_train_data
from common.logger import logger


# video_db_path = "/Volumes/Seagate Expansion Drive/byte/track2/video.db"
# title_feature_path = "/Volumes/Seagate Expansion Drive/byte/track2/title.db"
# user_db_path = "/Volumes/Seagate Expansion Drive/byte/track2/user.db"
deep_fm = DeepFM(10, 140000, [80000, 400, 900000, 500, 10, 90000, 80000, 30, 20, 500000], 128, 128,
                 embedding_size=128, learning_rate=0.003, use_bert=False, num_attention_heads=8,
                 batch_size=256, deep_layers_activation='sigmoid')
# exit()

"""
    train model
"""
model = deep_fm.train()
# model_path = '/home/yuanjun/code/Bytedance_ICME_challenge/track2/models/20190304/byte_305000.model'
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

total_epochs = 5

video_path = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/track2_video_features.txt"
title_path = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/track2_title.txt"
interactive_file = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/final_track2_train.txt"
audio_file_path = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/track2_audio_features.txt"
data_prepro_tool = DataPreprocessor()
iter_data = data_prepro_tool.get_train_data_from_origin_file(video_path, title_path, interactive_file, audio_file_path)
for epoch in range(total_epochs):

    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$epoch: %s$$$$$$$$$$$$$$$$$$$$$$$$$$$" % epoch)
    logger.info("$$$$$$$$$$$$$$$$$$$$$$$$$$$epoch: %s$$$$$$$$$$$$$$$$$$$$$$$$$$$" % epoch)
    for item in iter_data:
        print("loading consume: %s" % (time.time() - load_data_time))
        train_result, val_result = item
        print("train_result len:", len(train_result["index"]))
        deep_fm.fit2(model, optimizer, criterion, train_result["index"], train_result["value"], train_result["video"],
                     train_result['audio'], train_result["title"], train_result["title_value"], train_result["like"],
                     train_result["finish"], count,
                     save_path="/home/yuanjun/code/Bytedance_ICME_challenge/track2/models/",
                     Xi_valid=val_result["index"], Xv_valid=val_result["value"], audio_feature_val=val_result["audio"],
                     y_like_valid=val_result["like"], y_finish_valid=val_result["finish"],
                     video_feature_val=val_result["video"], title_feature_val=val_result["title"],
                     title_value_val=val_result["title_value"], total_epochs=total_epochs)
        count += 1
        load_data_time = time.time()
    iter_data = data_prepro_tool.get_train_data_from_origin_file(video_path, title_path, interactive_file, audio_file_path)

