from data_io.data_preprocessor import DataPreprocessor
# from model_zoo.din_xdeep_bert import DeepFM
from model_zoo.bert import DeepFM
from data_analy.user_interactive import UserInteractiveTool
from torch.optim import SparseAdam
from model_zoo.sparse_adam import SparseAdam
import torch
import time
from model_zoo.focal_loss import FocalLoss
from common.logger import logger
import datetime
from model_zoo.optimization import *


# video_db_path = "/Volumes/Seagate Expansion Drive/byte/track2/video.db"
# title_feature_path = "/Volumes/Seagate Expansion Drive/byte/track2/title.db"
# user_db_path = "/Volumes/Seagate Expansion Drive/byte/track2/user.db"
deep_fm = DeepFM(
    10, 140000, [80000, 400, 900000, 500, 10, 90000, 80000, 30, 20, UserInteractiveTool.ITEM_EMBEDDING_SIZE], 128, 128, bert_use_dropout=True,
    embedding_size=80, learning_rate=0.001, use_bert=True, use_cin=True, use_deep=True, num_attention_heads=8, bert_dropouts=0.1,
    batch_size=2048, weight_decay=0.0, deep_layers_activation='relu', is_shallow_dropout=False, is_batch_norm=False, use_line=True)

"""
    train model
"""
model = deep_fm.train()
# model_path = '/home/yuanjun/code/Bytedance_ICME_challenge/track2/models/like/byte_240000.model'
# deep_fm.load_state_dict(torch.load(model_path))
model.cuda(0)

# criterion = FocalLoss(2)
criterion = torch.nn.CrossEntropyLoss()

# print(len(optimizer.param_groups))
# param_optimizer = list(model.named_parameters())

# no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', "embeddings"]
# # no_decay = ["embeddings"]
# optimizer_grouped_parameters = [
#     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': model.weight_decay},
#     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
# ]
sparse_params = list()
dense_params = list()
for name, para in model.named_parameters():
    if name.find("fm_first_order_embeddings") != -1 or name.find("fm_second_order_embeddings") != -1:
        print("sparse tensor", name)
        sparse_params.append({'params': para, 'weight_decay': 0.1})
    else:
        dense_params.append({'params': para, 'weight_decay': 0.1})
# for i in optimizer_grouped_parameters:
#     # for name in i["params"]:
#     #     print(name)
#     print(len(i["params"]))
#     print(i["weight_decay"])
#     print("*****************************************************")
# exit()
# # for name, param in list(model.named_parameters()):
# #     print(name)
# exit()


count = 0
load_data_time = time.time()
total_epochs = 40
t_total = 20000000*total_epochs/model.batch_size
# optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate, weight_decay=model.weight_decay)

# optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=model.learning_rate, weight_decay=model.weight_decay)
# if model.optimizer_type == 'adam':
#     #
#     optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=model.learning_rate, weight_decay=model.weight_decay)
#     # optimizer = torch.optim.Adam([
#     #             {'params': model.base.parameters()},
#     #             {'params': model.title_embedding.parameters(), 'weight_decay': 1e-8},
#     #             {'params': model.fm_first_order_embeddings.parameters(), 'weight_decay': 1e-8},
#     #             {'params': model.fm_second_order_embeddings.parameters(), 'weight_decay': 1e-8}
#     #         ], lr=model.learning_rate, weight_decay=model.weight_decay)
# elif model.optimizer_type == 'rmsp':
#     optimizer = torch.optim.RMSprop(model.parameters(), lr=model.learning_rate, weight_decay=model.weight_decay)
# elif model.optimizer_type == 'adag':
#     optimizer = torch.optim.Adagrad(model.parameters(), lr=model.learning_rate, weight_decay=model.weight_decay)

video_path = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/track2_video_features.txt"
title_path = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/track2_title.txt"
interactive_file = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/final_track2_train.txt"
audio_file_path = "/home/yuanjun/code/Bytedance_ICME_challenge/track2/track2_audio_features.txt"
data_prepro_tool = DataPreprocessor()
iter_data = data_prepro_tool.get_train_data_from_origin_file(video_path, title_path, interactive_file, audio_file_path)
for epoch in range(total_epochs):
    optimizer = BertAdam(dense_params, lr=model.learning_rate*(0.5**epoch), warmup=0.002, t_total=t_total)
    optimizer1 = SparseAdam(sparse_params, lr=model.learning_rate*(0.5**epoch))

    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$epoch: %s$$$$$$$$$$$$$$$$$$$$$$$$$$$" % epoch)
    log_json = {"epoch": epoch, "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    logger.info("$$$$$$$$$$$$$$$$$$$$$$$$$$$epoch: %s$$$$$$$$$$$$$$$$$$$$$$$$$$$" % epoch)
    for item in iter_data:
        print("loading consume: %s" % (time.time() - load_data_time))
        train_result, val_result = item
        print("train_result len:", len(train_result["index"]), len(val_result["index"]))
        print("data check index:",  train_result["index"][0])
        print("data check value:",  train_result["value"][0])
        print("data check video:",  train_result["video"][0])
        print("data check audio:",  train_result['audio'][0])
        print("data check title:",  train_result["title"][0])
        print("data check title value:",  train_result["title_value"][0])
        print("data check like:",  train_result["like"][0])
        print("data check finish:",  train_result["finish"][0])

        deep_fm.fit2(model, [optimizer, optimizer1], criterion, train_result["index"], train_result["value"], train_result["video"],
                     train_result['audio'], train_result["title"], train_result["title_value"], train_result["like"],
                     train_result["finish"], count,
                     save_path="/home/yuanjun/code/Bytedance_ICME_challenge/track2/models/",
                     Xi_valid=val_result["index"], Xv_valid=val_result["value"], audio_feature_val=val_result["audio"],
                     y_like_valid=val_result["like"], y_finish_valid=val_result["finish"],
                     video_feature_val=val_result["video"], title_feature_val=val_result["title"],
                     title_value_val=val_result["title_value"], total_epochs=total_epochs, current_epoch=epoch, task="finish")
        count += 1
        load_data_time = time.time()
    iter_data = data_prepro_tool.get_train_data_from_origin_file(video_path, title_path, interactive_file, audio_file_path)

