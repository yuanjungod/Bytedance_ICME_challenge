# from model_zoo.din_xdeep_bert import DeepFM
from model_zoo.bert import DeepFM
from data_analy.user_interactive import UserInteractiveTool
import time
from model_zoo.focal_loss import FocalLoss
from common.logger import logger
import datetime
from model_zoo.optimization import *
from data_io.data_preprocessor import DataPreprocessor
from data_io.data_set import MyDataSet
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import numpy as np
from sklearn.metrics import roc_auc_score
from model_zoo.sparse_adam import SparseAdam


deep_fm = DeepFM(
    10, 140000, [80000, 400, 900000, 500, 10, 90000, 80000, 30, 20, UserInteractiveTool.ITEM_EMBEDDING_SIZE], 128, 128, bert_use_dropout=True,
    embedding_size=80, learning_rate=0.001, use_bert=True, use_cin=True, use_deep=True, num_attention_heads=8, bert_dropouts=0.5,
    batch_size=2048, weight_decay=0.0, deep_layers_activation='relu', is_shallow_dropout=False, is_batch_norm=False, use_line=True)

loader = DataPreprocessor()
loader.get_origin_train_data()

train_data = MyDataSet(loader)
train_dataloader = DataLoader(train_data, batch_size=2048*10, shuffle=True, num_workers=4)

test_data = MyDataSet(loader, train=False)
test_loader = DataLoader(test_data, batch_size=2048, shuffle=True, num_workers=4)
epochs = 40
model = deep_fm.train()
model.cuda(0)

# criterion = torch.nn.CrossEntropyLoss()
criterion = FocalLoss()
t_total = 20000000*epochs/model.batch_size

sparse_params = list()
dense_params = list()
for name, para in model.named_parameters():
    if name.find("fm_first_order_embeddings") != -1 or name.find("fm_second_order_embeddings") != -1:
        print("sparse tensor", name)
        sparse_params.append({'params': para, 'weight_decay': 0.1})
    else:
        dense_params.append({'params': para, 'weight_decay': 0.1})

for epoch in trange(int(epochs), desc="Epoch"):

    optimizer = BertAdam(dense_params, lr=model.learning_rate * (0.1 ** epoch), warmup=0.002, t_total=t_total)
    optimizer1 = SparseAdam(sparse_params, lr=model.learning_rate * (0.1 ** epoch))

    for step, train_result_list in enumerate(tqdm(train_dataloader, desc="Iteration")):
        # print(len(train_result_list[2]))
        # print([i.numpy().shape for i in train_result_list[2]])
        train_result = {
            'like': train_result_list[0].numpy(),
            'finish': train_result_list[1].numpy(),
            'index': [i.numpy() for i in train_result_list[2]],
            'value': [i.numpy() for i in train_result_list[3]],
            'title': [i.numpy() for i in train_result_list[4]],
            'title_value': [i.numpy() for i in train_result_list[5]],
            'item_id': train_result_list[6].numpy(),
            "video": [i.numpy() for i in train_result_list[7]],
            "audio": [i.numpy() for i in train_result_list[8]],
            'feature_sizes': loader.FEATURE_SIZES,
            'tile_word_size': loader.title_feature_tool.MAX_WORD}

        deep_fm.fit2(model, [optimizer, optimizer1], criterion, train_result["index"], train_result["value"],
                     train_result["video"],
                     train_result['audio'], train_result["title"], train_result["title_value"], train_result["like"],
                     train_result["finish"], step,
                     save_path="/home/yuanjun/code/Bytedance_ICME_challenge/track2/models/",
                     total_epochs=epochs, current_epoch=epoch,
                     task="finish")

        if (step+1) % 250 == 0:
            like_preb_list = list()
            finish_preb_list = list()
            like_list = list()
            finish_list = list()
            for val_result_list in tqdm(test_loader, desc="Evaluating"):

                val_result = {
                    'like': val_result_list[0].numpy(),
                    'finish': val_result_list[1].numpy(),
                    'index': [i.numpy() for i in val_result_list[2]],
                    'value': [i.numpy() for i in val_result_list[3]],
                    'title': [i.numpy() for i in val_result_list[4]],
                    'title_value': [i.numpy() for i in val_result_list[5]],
                    'item_id': val_result_list[6].numpy(),
                    "video": [i.numpy() for i in val_result_list[7]],
                    "audio": [i.numpy() for i in val_result_list[8]],
                    'feature_sizes': loader.FEATURE_SIZES,
                    'tile_word_size': loader.title_feature_tool.MAX_WORD}

                with torch.no_grad():
                    like_preb, finish_preb = model.predict_proba(
                        val_result["index"], val_result["value"], val_result["video"],
                        val_result["audio"], val_result["title"], val_result["title_value"])

                    like_preb = like_preb[:, 1]
                    finish_preb = finish_preb[:, 1]

                    like_preb_list.append(like_preb)
                    finish_preb_list.append(finish_preb)

                    like_list.append(val_result["like"])
                    finish_list.append(val_result["finish"])

            logger.info("like auc: %s" % roc_auc_score(
                np.concatenate(np.array(like_list), 0), np.concatenate(np.array(like_preb_list), 0)))
            logger.info("finish auc: %s" % roc_auc_score(
                np.concatenate(np.array(finish_list), 0), np.concatenate(np.array(finish_preb_list), 0)))




