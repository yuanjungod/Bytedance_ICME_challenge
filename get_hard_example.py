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
from data_analy.title_analy import TitleAnalyTool
import json

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

video_db_path = "/Volumes/Seagate Expansion Drive/byte/track2/video.db"
title_feature_path = "/Volumes/Seagate Expansion Drive/byte/track2/title.db"
user_db_path = "/Volumes/Seagate Expansion Drive/byte/track2/user.db"
# video_db_path = "/Volumes/Seagate Expansion Drive/byte/track2/video.db"
# title_feature_path = "/Volumes/Seagate Expansion Drive/byte/track2/title.db"
# user_db_path = "/Volumes/Seagate Expansion Drive/byte/track2/user.db"
video_path = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/track2_video_features.txt"
title_path = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/track2_title.txt"
interactive_file = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/final_track2_train.txt"
audio_file_path = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/track2_audio_features.txt"
data_prepro_tool = DataPreprocessor()
hard_result = {'like': [], 'finish': [], 'index': [], 'value': [], 'title': [], 'title_value': [], 'item_id': [],
                "video": [], 'audio': [], 'feature_sizes': DataPreprocessor.FEATURE_SIZES,
                'tile_word_size': TitleAnalyTool.MAX_WORD}
for result in data_prepro_tool.get_train_data():
    like_preb, finish_preb = deep_fm.predict_proba(
        result["index"], result["value"], result["video"], result['audio'], result["title"], result["title_value"])
    for i in range(len(like_preb[:, 1])):
        if abs(result["like"] - like_preb[:, 1]) > 0.7 or abs(result["finish"] - finish_preb[:, 1]) > 0.7:
            hard_result['like'].append(result['like'])
            hard_result['finish'].append(result['finish'])
            hard_result['index'].append(result['index'])
            hard_result['value'].append(result['value'])
            hard_result['title'].append(result['title'])
            hard_result['title_value'].append(result['title_value'])
            hard_result['item_id'].append(result['item_id'])
            hard_result['video'].append(result['video'])
            hard_result['audio'].append(result['audio'])
hard_file = open("hard.json", "w")
json.dump(hard_result, hard_file)
hard_file.close()

