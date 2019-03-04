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
import numpy as np
from sklearn.metrics import roc_auc_score
import traceback
import argparse


parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument("--debug", action='store_true')
debug = parser.parse_args().debug
print(debug)
# exit()

# 通过下面的方式进行简单配置输出方式与日志级别

# video_db_path = "/Volumes/Seagate Expansion Drive/byte/track2/video.db"
# title_feature_path = "/Volumes/Seagate Expansion Drive/byte/track2/title.db"
# user_db_path = "/Volumes/Seagate Expansion Drive/byte/track2/user.db"
video_db_path = "/Volumes/Seagate Expansion Drive/byte/track2/video.db"
title_feature_path = "/Volumes/Seagate Expansion Drive/byte/track2/title.db"
user_db_path = "/Volumes/Seagate Expansion Drive/byte/track2/user.db"
deep_fm = DeepFM(10, 140000, [80000, 400, 900000, 500, 10, 90000, 80000, 30, 20, 500000], 128, 128,
                 embedding_size=128, learning_rate=0.003, use_bert=True, num_attention_heads=8, batch_size=128)
# exit()


"""
    train model
"""
model = deep_fm.train()
# model_path = '/Volumes/Seagate Expansion Drive/byte/track2/models/20190304/byte_305000.model'
model_path = '/home/yuanjun/code/Bytedance_ICME_challenge/track2/models/20190304/byte_305000.model'
# deep_fm.load_state_dict(torch.load(model_path, map_location='cpu'))
deep_fm.load_state_dict(torch.load(model_path))
model.cuda(0)

load_data_time = time.time()

# video_path = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/video.db"
# title_path = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/title.db"
# interactive_file = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/user.db"
# audio_file_path = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/audio.db"
if debug:
    video_path = "/Volumes/Seagate Expansion Drive/byte/track2/video.db"
    title_path = "/Volumes/Seagate Expansion Drive/byte/track2/title.db"
    interactive_file = "/Volumes/Seagate Expansion Drive/byte/track2/user.db"
    audio_file_path = "/Volumes/Seagate Expansion Drive/byte/track2/audio.db"
    data_prepro_tool = DataPreprocessor(video_path, interactive_file, title_path, audio_file_path)
    result_list = data_prepro_tool.get_train_data(100)
else:
    video_path = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/track2_video_features.txt"
    title_path = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/track2_title.txt"
    interactive_file = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/final_track2_train.txt"
    audio_file_path = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/track2_audio_features.txt"
    data_prepro_tool = DataPreprocessor(video_path, interactive_file, title_path, audio_file_path)
    result_list = data_prepro_tool.get_train_data_from_origin_file(video_path, title_path, interactive_file, audio_file_path, 256)


like_preb_list = list()
finish_preb_list = list()
like_list = list()
finish_list = list()
total_count = 0

for train_result, val_result in result_list:
    result = train_result
    total_count += len(result["index"])
    like_preb, finish_preb = deep_fm.predict_proba(
        result["index"], result["value"], result["video"], result['audio'],  result["title"], result["title_value"])
    like_preb_list.append(like_preb[:, 1])
    finish_preb_list.append(finish_preb[:, 1])
    like_list.append(np.array(result["like"]))
    finish_list.append(np.array(result["finish"]))
    load_data_time = time.time()
    try:
        print("############################sample count: %s############################" % total_count)
        print("like auc: ", roc_auc_score(
            np.concatenate(np.array(like_list), 0), np.concatenate(np.array(like_preb_list), 0)))
        print("finish auc: ", roc_auc_score(
            np.concatenate(np.array(finish_list), 0), np.concatenate(np.array(finish_preb_list), 0)))
    except:
        print(traceback.format_exc())




