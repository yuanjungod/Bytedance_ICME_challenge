# -*- coding:utf-8 -*-

"""
Created on Dec 10, 2017
@author: jachin,Nie

A pytorch implementation of deepfm

Reference:
[1] DeepFM: A Factorization-Machine based Neural Network for CTR Prediction,
    Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.

"""

import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from time import time

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utils.utils import warmup_linear
import torch.backends.cudnn
from .interest_pooling import InterestPooling1
from .bert_model import BertConfig, BertModel
from common.logger import logger
import json
import datetime
import traceback
import random
from .dice import *

"""
    网络结构部分
"""


class DeepFM(torch.nn.Module):
    """
    :parameter
    -------------
    field_size: size of the feature fields
    feature_sizes: a field_size-dim array, sizes of the feature dictionary
    embedding_size: size of the feature embedding
    is_shallow_dropout: bool, shallow part(fm or ffm part) uses dropout or not?
    dropout_shallow: an array of the size of 2, example:[0.5,0.5], the first element is for the-first order part and the second element is for the second-order part
    h_depth: deep network's hidden layers' depth
    deep_layers: a h_depth-dim array, each element is the size of corresponding hidden layers. example:[32,32] h_depth = 2
    is_deep_dropout: bool, deep part uses dropout or not?
    dropout_deep: an array of dropout factors,example:[0.5,0.5,0.5] h_depth=2
    deep_layers_activation: relu or sigmoid etc
    n_epochs: epochs
    batch_size: batch_size
    learning_rate: learning_rate
    optimizer_type: optimizer_type, 'adam', 'rmsp', 'sgd', 'adag'
    is_batch_norm：bool,  use batch_norm or not ?
    verbose: verbose
    weight_decay: weight decay (L2 penalty)
    random_seed: random_seed=950104 someone's birthday, my lukcy number
    use_fm: bool
    use_ffm: bool
    use_deep: bool
    loss_type: "logloss", only
    eval_metric: roc_auc_score
    use_cuda: bool use gpu or cpu?
    n_class: number of classes. is bounded to 1
    greater_is_better: bool. Is the greater eval better?


    Attention: only support logsitcs regression
    """

    def __init__(self, field_size, tile_word_size, feature_sizes, video_feature_size, audio_feature_size,
                 embedding_size=32, is_shallow_dropout=True, dropout_shallow=[0.5, 0.5],
                 h_depth=2, deep_layers=[128, 64], is_deep_dropout=True, dropout_deep=[0.5, 0.5, 0.5],
                 deep_layers_activation='relu', n_epochs=64, batch_size=256, learning_rate=0.003,
                 optimizer_type='adam', is_batch_norm=False, verbose=True, random_seed=950104, weight_decay=0.0,
                 use_fm=True, use_ffm=False, use_deep=True, loss_type='logloss', eval_metric=roc_auc_score,
                 use_cuda=True, n_class=2, greater_is_better=True, cin_deep_layers=[100, 32], cin_layer_sizes=[50, 50, 50, 100],
                 cin_activation='relu', use_line=False,
                 is_cin_bn=True,
                 cin_direct=False, use_cin_bias=False, bert_use_dropout=False, num_hidden_layers=12,
                 cin_deep_dropouts=[0.5, 0.5], use_cin=True, use_bert=True, bert_dropouts=0.5, num_attention_heads=8
                 ):
        super(DeepFM, self).__init__()
        self.use_line = use_line
        self.total_count = 0
        self.field_size = field_size
        self.feature_sizes = feature_sizes
        self.tile_word_size = tile_word_size
        self.embedding_size = embedding_size
        self.video_feature_size = video_feature_size
        self.is_shallow_dropout = is_shallow_dropout
        self.dropout_shallow = dropout_shallow
        self.h_depth = h_depth
        self.deep_layers = deep_layers
        self.is_deep_dropout = is_deep_dropout
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.is_batch_norm = is_batch_norm
        self.verbose = verbose
        self.weight_decay = weight_decay
        self.random_seed = random_seed
        self.use_fm = use_fm
        self.use_ffm = use_ffm
        self.use_deep = use_deep
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.use_cuda = use_cuda
        self.n_class = n_class
        self.greater_is_better = greater_is_better

        self.cin_activation = cin_activation
        self.is_cin_bn = is_cin_bn
        self.cin_layer_sizes = [self.field_size+3] + cin_layer_sizes
        self.cin_direct = cin_direct
        self.use_cin_bias = use_cin_bias

        self.use_cin = use_cin
        self.cin_deep_layers = cin_deep_layers
        self.cin_deep_dropouts = cin_deep_dropouts
        self.bert_use_dropout = bert_use_dropout
        self.use_bert = use_bert
        self.num_attention_heads = num_attention_heads

        torch.manual_seed(self.random_seed)

        """
            check cuda
        """
        if self.use_cuda and not torch.cuda.is_available():
            self.use_cuda = False
            print("Cuda is not available, automatically changed into cpu model")

        """
          InterestPooling1
        """
        self.interest_pooling = InterestPooling1(30, embedding_size, use_cuda=self.use_cuda)

        """
            bias
        """
        if self.use_fm or self.use_ffm:
            self.bias = torch.nn.Parameter(torch.randn(1))
        """
            fm part
        """
        self.title_embedding = nn.Embedding(tile_word_size, embedding_size)
        self.video_line = nn.Linear(video_feature_size, embedding_size)
        self.video_dice = Dice(self.embedding_size)
        self.audio_line = nn.Linear(audio_feature_size, embedding_size)
        self.audio_dice = Dice(self.embedding_size)
        self.bert_dice = Dice(self.embedding_size)
        print("Init fm part")
        if self.use_line:
            self.fm_first_order_embeddings = nn.ModuleList(
                [nn.Embedding(feature_size, 1, sparse=True) for feature_size in self.feature_sizes])
            if self.dropout_shallow:
                self.fm_first_order_dropout = nn.Dropout(self.dropout_shallow[0])
        self.fm_second_order_embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, self.embedding_size, sparse=True) for feature_size in self.feature_sizes])
        if self.dropout_shallow:
            self.fm_second_order_dropout = nn.Dropout(self.dropout_shallow[1])
        print("Init fm part succeed")

        logger.info(json.dumps({"init info": "Init bert part"}))
        self.config = BertConfig(
            hidden_size=embedding_size, num_hidden_layers=num_hidden_layers, attention_probs_dropout_prob=0.1,
            use_index=[-1, 0, 1, 4, 7, 10], num_attention_heads=num_attention_heads, intermediate_size=embedding_size*4)
        self.bert_model = BertModel(self.config, self.use_cuda)
        logger.info(json.dumps({"init info": "Init bert part succeed"}))
        """
            linear_layer
        """
        self.bert_line = nn.Linear(self.embedding_size*len(self.config.use_index), self.embedding_size)
        concat_input_size = self.embedding_size
        if self.use_line:
            concat_input_size += self.field_size
        if self.use_bert:
            concat_input_size += self.embedding_size
        if self.bert_use_dropout:
            self.bert_drop_out = nn.Dropout(bert_dropouts)
        # self.like_concat_linear_layer = nn.Linear(concat_input_size, 64)

        self.like_concat_linear_layer1 = nn.Linear(concat_input_size, self.n_class)

        # self.finish_concat_linear_layer = nn.Linear(concat_input_size, 64)
        self.finish_concat_linear_layer2 = nn.Linear(concat_input_size, self.n_class)

        # self.result_drop_out = nn.Dropout(0.8)

        print("Init succeed")

    def forward(self, Xi, Xv, video_feature, audio_feature, title_feature, title_value):
        """
        :param Xi_train: index input tensor, batch_size * k * 1
        :param Xv_train: value input tensor, batch_size * k * 1
        :return: the last output
        """
        if self.use_line:
            fm_first_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in
                                      enumerate(self.fm_first_order_embeddings)]
            # fm_first_order_emb_arr = list()
            # for i, emb in enumerate(self.fm_first_order_embeddings):
            #     # print(Xi[:, i, :], emb)
            #     print(Xi[:, i, :].size(), emb, torch.sum(emb(Xi[:, i, :]), 1).t())
            #     fm_first_order_emb_arr.append((torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t())
            # print([i.size() for i in fm_first_order_emb_arr])
            fm_first_order = torch.cat(fm_first_order_emb_arr, 1)
            if self.is_shallow_dropout:
                fm_first_order = self.fm_first_order_dropout(fm_first_order)

        # use 2xy = (x+y)^2 - x^2 - y^2 reduce calculation
        fm_second_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in
                                   enumerate(self.fm_second_order_embeddings)]

        fm_sum_second_order_emb = sum(fm_second_order_emb_arr)
        # print("sum", fm_sum_second_order_emb.size())
        # exit()
        fm_sum_second_order_emb_square = fm_sum_second_order_emb * fm_sum_second_order_emb  # (x+y)^2
        fm_second_order_emb_square = [item * item for item in fm_second_order_emb_arr]
        fm_second_order_emb_square_sum = sum(fm_second_order_emb_square)  # x^2+y^2
        fm_second_order = (fm_sum_second_order_emb_square - fm_second_order_emb_square_sum) * 0.5
        fm_second_order = self.fm_second_order_dropout(fm_second_order)

        if self.deep_layers_activation == 'sigmoid':
            activation = torch.sigmoid
        elif self.deep_layers_activation == 'tanh':
            activation = F.tanh
        else:
            activation = F.relu

        # if self.use_line:
        #     deep_emb = torch.cat([fm_first_order, fm_second_order_emb_arr, fm_second_order], 1)
        # else:
        #     deep_emb = torch.cat([fm_second_order_emb_arr, fm_second_order], 1)
        deep_emb = torch.cat(fm_second_order_emb_arr, 1)
        if video_feature is not None:

            if self.embedding_size != 128:
                video_feature = self.video_line(video_feature)
                # video_feature = activation(video_feature)
                video_feature = self.video_dice(video_feature)

            deep_emb = torch.cat([deep_emb, video_feature], 1)

        if audio_feature is not None:
            if self.embedding_size != 128:
                audio_feature = self.audio_line(audio_feature)
                # audio_feature = activation(audio_feature)
                audio_feature = self.audio_dice(audio_feature)

            deep_emb = torch.cat([deep_emb, audio_feature], 1)

        title_embedding = self.title_embedding(title_feature)
        title_size = title_embedding.size()
        title_embedding = title_embedding.view(-1, title_size[1] * title_size[2])
        title_embedding = self.interest_pooling(title_embedding, title_value)

        deep_emb = torch.cat([deep_emb, title_embedding], 1)

        bert_emb_size = deep_emb.size()
        label = Variable(torch.zeros(bert_emb_size[0], 1, dtype=torch.long))
        if self.use_cuda:
            label = label.cuda()

        bert_emb = deep_emb.view(bert_emb_size[0], -1, self.embedding_size)

        bert_result = self.bert_model(bert_emb, label)
        # print(bert_result.size())
        # exit()
        if self.bert_use_dropout:
            bert_result = self.bert_drop_out(bert_result)

        # bert_result = bert_result.view(bert_emb_size[0], -1)
        bert_result = self.bert_line(bert_result)
        # bert_result = activation(bert_result)
        bert_result = self.bert_dice(bert_result)

        if self.use_line:
            concat_input = torch.cat([fm_first_order, bert_result, fm_second_order], 1)
        else:
            concat_input = torch.cat([bert_result, fm_second_order], 1)

        like = self.like_concat_linear_layer1(concat_input)

        finish = self.finish_concat_linear_layer2(concat_input)

        return like, finish

    def fit2(self, model, optimizer, criterion, Xi_train, Xv_train, video_feature, audio_feature, title_feature, title_value,
             y_like_train, y_finish_train, count, Xi_valid=None, Xv_valid=None, y_like_valid=None, y_finish_valid=None,
             video_feature_val=None, audio_feature_val=None, title_feature_val=None,
             title_value_val=None, save_path=None, total_epochs=3, current_epoch=0, task="finish"):

        # if save_path and not os.path.exists('/'.join(save_path.split('/')[0:-1])):
        #     print("Save path is not existed!")
        #     return
        self.train()

        Xi_train = np.array(Xi_train).T.reshape((-1, self.field_size, 1))
        video_feature = np.array(video_feature).T
        audio_feature = np.array(audio_feature).T
        title_feature = np.array(title_feature).T
        title_value = np.array(title_value).T
        Xv_train = np.array(Xv_train).T

        # title_feature = np.array(title_feature)
        # title_value = [[[j for _ in range(self.embedding_size)] for j in i] for i in title_value]
        # title_value = np.array(title_value)

        y_like_train = np.array(y_like_train)
        y_finish_train = np.array(y_finish_train)
        # y_train = np.concatenate([y_like_train.reshape(-1, 1), y_finish_train.reshape(-1, 1)], 1)
        x_size = Xi_train.shape[0]

        # if self.verbose:
        #     print("pre_process data finished")
        #     print("Xi_train", Xi_train.shape)
        #     print("Xv_train", Xv_train.shape)
        #     print("video_feature", video_feature.shape)
        #     print("audio_feature", audio_feature.shape)
        #     print("title_feature", title_feature.shape)
        #     print("title_value", title_value.shape)
        #     print("y_like_train", y_like_train.shape)
        #     print("y_finish_train", y_finish_train.shape)

        total_loss = 0.0
        batch_iter = x_size // self.batch_size
        epoch_begin_time = time()
        batch_begin_time = time()
        # current_learn_rate = 0
        for i in range(batch_iter + 1):
            self.total_count += 1
            offset = i * self.batch_size
            end = min(x_size, offset + self.batch_size)
            if offset == end:
                break
            batch_xi = Variable(torch.LongTensor(Xi_train[offset:end]))
            batch_xv = Variable(torch.FloatTensor(Xv_train[offset:end]))
            batch_y_like_train = torch.LongTensor(y_like_train[offset:end])
            batch_y_finish_train = torch.LongTensor(y_finish_train[offset:end])
            # weight_list = list()
            if task == "finish":
                ratio = int(len(y_finish_train[offset:end])/(sum(y_finish_train[offset:end])+1))
            else:
                ratio = int(len(y_like_train[offset:end]) / (sum(y_like_train[offset:end]) + 1))
            if random.random() > 0.99:
                # print("ratio", ratio)
                pass
            weight_list = [1, ratio-1]
            # for i in y_finish_train[offset:end]:
            #     if i == 0:
            #         weight_list.append(1)
            #     else:
            #         weight_list.append(ratio-1)
            batch_weight_list = torch.FloatTensor(weight_list)
            # batch_label = Variable(torch.cat([torch.FloatTensor(y_like_train[offset:end]).view(-1, 1),
            #                                   torch.FloatTensor(y_finish_train[offset:end]).view(-1, 1)], -1))
            try:
                batch_video_feature = Variable(torch.FloatTensor(video_feature[offset:end]))
            except:
                print(len(video_feature[offset:end]))
                print([len(i) for i in video_feature[offset:end]])
                traceback.print_exc()
                exit()
            try:
                batch_audio_feature = Variable(torch.FloatTensor(audio_feature[offset:end]))
            except:
                print(len([len(i) for i in audio_feature[offset:end]]))
                print([len(i) for i in audio_feature[offset:end]])
                traceback.print_exc()
                exit()
            batch_title_value = Variable(torch.FloatTensor(title_value[offset:end]))
            batch_title_feature = Variable(torch.LongTensor(title_feature[offset:end]))

            if self.use_cuda:
                batch_xi, batch_xv, batch_y_like_train, batch_y_finish_train, batch_video_feature, batch_audio_feature, batch_title_value, batch_title_feature, batch_weight_list = \
                    batch_xi.cuda(), batch_xv.cuda(), batch_y_like_train.cuda(), batch_y_finish_train.cuda(), batch_video_feature.cuda(), \
                    batch_audio_feature.cuda(), batch_title_value.cuda(), batch_title_feature.cuda(), batch_weight_list.cuda()

            for item in optimizer:
                item.zero_grad()

            outputs = model(batch_xi, batch_xv, batch_video_feature, batch_audio_feature, batch_title_feature, batch_title_value)
            # print("outputs size: ", outputs.size())
            like, finish = outputs
            # print("like", like.size())
            # print("batch_y_like_train", batch_y_like_train.size())
            # criterion.weight = batch_weight_list
            like_loss = criterion(like, batch_y_like_train)
            finish_loss = criterion(finish, batch_y_finish_train)
            loss = like_loss + 0.2*finish_loss
            # if task == "finish":
            #     loss = finish_loss
            # else:
            #     loss = like_loss
            # loss.backward()
            # loss = like_loss
            loss.backward()
            # print(loss)
            # exit()

            # for param_group in optimizer.param_groups:
            #     current_learn_rate = self.learning_rate*warmup_linear(self.total_count/(
            #             20000000*total_epochs/self.batch_size))
            #     param_group['lr'] = current_learn_rate

            for item in optimizer:
                item.step()
            # print(loss)
            total_loss += loss.data
            if self.verbose:
                if self.total_count % 100 == 0:  # print every 100 mini-batches
                    # eval = self.evaluate(batch_xi, batch_xv, batch_video_feature, batch_audio_feature,
                    # batch_title_feature, batch_title_value, batch_y_like_train, batch_y_finish_train)
                    try:
                        like_auc = self.eval_metric(batch_y_like_train.cpu().detach().numpy(), F.softmax(like, dim=-1).cpu().detach().numpy()[:, 1])
                        finish_auc = self.eval_metric(batch_y_finish_train.cpu().detach().numpy(), F.softmax(finish, dim=-1).cpu().detach().numpy()[:, 1])
                        # print('****train***[%d, %5d] metric: like-%.6f, finish-%.6f, learn rate: %s, time: %.1f s' %
                        #       (count + 1, i + 1, like_auc, finish_auc, 0,
                        #        time() - batch_begin_time))

                        log_json = {"timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "status": "train", "count": count + 1,
                                    "loss": "%s" % (total_loss/100),
                                    "like_auc": "%s" % like_auc,
                                    "finish_auc": "%s" % finish_auc,
                                    "current_learn_rate": 0,
                                    "time": time() - epoch_begin_time}
                        logger.info(json.dumps(log_json))
                    except:
                        print("eval wrong!!!!!!")

                    total_loss = 0.0
                    batch_begin_time = time()
            # if self.total_count % 100 == 0:
            #     print("total count", self.total_count)
            if save_path and self.total_count % 2000 == 0:
                torch.save(self.state_dict(), os.path.join(save_path, "byte_%s.model" % self.total_count))

        if Xi_valid is not None and count % 20 == 0:
            print('valid*' * 20)
            Xi_valid = np.array(Xi_valid).reshape((-1, self.field_size, 1))
            Xv_valid = np.array(Xv_valid)

            title_feature_val = np.array(title_feature_val)
            # title_value_val = [[[j for _ in range(self.embedding_size)] for j in i] for i in title_value_val]
            title_value_val = np.array(title_value_val)
            video_feature_val = np.array(video_feature_val)
            audio_feature_val = np.array(audio_feature_val)
            y_like_valid = np.array(y_like_valid)
            y_finish_valid = np.array(y_finish_valid)
            y_valid = np.concatenate([y_like_valid.reshape(-1, 1), y_finish_valid.reshape(-1, 1)], 1)
            x_valid_size = Xi_valid.shape[0]
            try:
                valid_loss, valid_eval = self.eval_by_batch(
                    Xi_valid, Xv_valid, y_like_valid, y_finish_valid, x_valid_size, video_feature_val,
                    audio_feature_val,
                    title_feature_val, title_value_val)
                # valid_result.append(valid_eval)

                print('val [%d] loss: %.6f metric: like-%.6f,finish-%.6f, learn rate: %s,  time: %.1f s' %
                      (count + 1, valid_loss, valid_eval[0], valid_eval[1], 0,
                       time() - epoch_begin_time))
                log_json = {"timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "status": "val", "count": count + 1, "loss": "%s" % valid_loss.data,
                            "like_auc": "%s" % valid_eval[0],
                            "finish_auc": "%s" % valid_eval[1], "current_learn_rate": 0,
                            "time": time() - epoch_begin_time}
                logger.info(json.dumps(log_json))
                print('valid*' * 20)
            except:
                traceback.print_exc()
                print("eval wrong!!!!!!")

    def eval_by_batch(self, Xi, Xv, like_y, finish_y, x_size, video_feature, audio_feature, title_feature, title_value):
        total_loss = 0.0
        like_y_pred = []
        finish_y_pred = []
        # y_pred = []
        if self.use_ffm:
            batch_size = 16384 * 2
        else:
            batch_size = 256
        batch_iter = x_size // batch_size
        # criterion = F.binary_cross_entropy_with_logits
        criterion = F.cross_entropy
        model = self.eval()
        for i in range(batch_iter + 1):
            offset = i * batch_size
            end = min(x_size, offset + batch_size)
            if offset == end:
                break
            batch_xi = Variable(torch.LongTensor(Xi[offset:end]))
            batch_xv = Variable(torch.FloatTensor(Xv[offset:end]))
            batch_like_y = Variable(torch.LongTensor(like_y[offset:end]))
            batch_finish_y = Variable(torch.LongTensor(finish_y[offset:end]))
            batch_video_feature = Variable(torch.FloatTensor(video_feature[offset:end]))
            batch_audio_feature = Variable(torch.FloatTensor(audio_feature[offset:end]))
            batch_title_feature = Variable(torch.LongTensor(title_feature[offset:end]))
            batch_title_value = Variable(torch.FloatTensor(title_value[offset:end]))
            if self.use_cuda:
                batch_xi, batch_xv, batch_like_y, batch_finish_y, batch_video_feature, batch_audio_feature, batch_title_feature, batch_title_value = \
                    batch_xi.cuda(), batch_xv.cuda(), batch_like_y.cuda(), batch_finish_y.cuda(), batch_video_feature.cuda(), \
                    batch_audio_feature.cuda(), batch_title_feature.cuda(), batch_title_value.cuda()
            outputs = model(batch_xi, batch_xv, batch_video_feature, batch_audio_feature, batch_title_feature, batch_title_value)
            # print("outputs", outputs)
            like, finish = outputs
            like_pred = F.softmax(like, dim=-1).cpu()
            finish_pred = F.softmax(finish, dim=-1).cpu()
            # pred = torch.sigmoid(outputs).cpu()
            like_y_pred.append(like_pred.data.numpy())
            finish_y_pred.append(finish_pred.data.numpy())
            # y_pred.append(pred.data.numpy())
            # print("eval like", like.size())
            # print("eval batch_like_y", batch_like_y.size())
            like_loss = criterion(like, batch_like_y)
            finish_loss = criterion(finish, batch_finish_y)
            loss = like_loss + finish_loss
            total_loss += loss.data * (end - offset)
        # y_pred = np.array(y_pred)
        # y_pred = np.concatenate(y_pred, 0)
        # like_y_pred
        like_y_pred = np.concatenate(like_y_pred, 0)
        finish_y_pred = np.concatenate(finish_y_pred, 0)
        # size = y_pred.shape
        # print("y_pred", y_pred.shape, y_pred[0].shape)
        # y_pred = y_pred.reshape(-1, 2)
        total_metric = [self.eval_metric(like_y, like_y_pred[:, 1]), self.eval_metric(finish_y, finish_y_pred[:, 1])]
        return total_loss / x_size, total_metric

    # shuffle three lists simutaneously
    def shuffle_in_unison_scary(self, a, b, c):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)

    def training_termination(self, valid_result):
        if len(valid_result) > 4:
            if self.greater_is_better:
                if valid_result[-1] < valid_result[-2] and valid_result[-2] < valid_result[-3] and \
                        valid_result[-3] < valid_result[-4]:
                    return True
            else:
                if valid_result[-1] > valid_result[-2] and \
                        valid_result[-2] > valid_result[-3] and \
                        valid_result[-3] > valid_result[-4]:
                    return True
        return False

    def predict(self, Xi, Xv, video_feature, title_feature, title_value):

        Xi = np.array(Xi).reshape((-1, self.field_size, 1))
        Xi = Variable(torch.LongTensor(Xi))
        Xv = Variable(torch.FloatTensor(Xv))
        if self.use_cuda and torch.cuda.is_available():
            Xi, Xv = Xi.cuda(), Xv.cuda()

        model = self.eval()
        pred = torch.sigmoid(model(Xi, Xv)).cpu()
        return pred

    def predict_proba(self, Xi, Xv, video_feature, audio_feature, title_feature, title_value):
        Xi = np.array(Xi).T.reshape((-1, self.field_size, 1))
        Xv = np.array(Xv).T
        video_feature = np.array(video_feature).T
        audio_feature = np.array(audio_feature).T
        title_feature = np.array(title_feature).T
        title_value = np.array(title_value).T

        Xi = Variable(torch.LongTensor(Xi))
        Xv = Variable(torch.FloatTensor(Xv))

        # video_feature = np.array(video_feature)
        video_feature = Variable(torch.FloatTensor(video_feature))

        # audio_feature = np.array(audio_feature)
        audio_feature = Variable(torch.FloatTensor(audio_feature))

        title_feature = np.array(title_feature)
        title_feature = Variable(torch.LongTensor(title_feature))

        # title_value = [[[j for _ in range(self.embedding_size)] for j in i] for i in title_value]
        title_value = np.array(title_value)
        title_value = Variable(torch.FloatTensor(title_value))

        if self.use_cuda:
            Xi, Xv, video_feature, audio_feature, title_value, title_feature = \
                Xi.cuda(), Xv.cuda(), video_feature.cuda(), audio_feature.cuda(),\
                title_value.cuda(), title_feature.cuda()

        model = self.eval()
        outputs = model(Xi, Xv, video_feature, audio_feature, title_feature, title_value)
        like, finish = outputs
        like_preb = F.softmax(like, dim=-1)
        finish_preb = F.softmax(finish, dim=-1)
        return like_preb.cpu().detach().numpy(), finish_preb.cpu().detach().numpy()

    def inner_predict(self, Xi, Xv):
        """
        :param Xi: tensor of feature index
        :param Xv: tensor of feature value
        :return: output, numpy
        """
        model = self.eval()
        pred = torch.sigmoid(model(Xi, Xv)).cpu()
        return (pred.data.numpy() > 0.5)

    def inner_predict_proba(self, Xi, Xv, video_feature, audio_feature, title_feature, title_value):
        """
        :param Xi: tensor of feature index
        :param Xv: tensor of feature value
        :return: output, numpy
        """
        model = self.eval()
        out_put = model(Xi, Xv, video_feature, audio_feature, title_feature, title_value)
        like_out_put, finish_out_put = out_put
        like_out_put = F.softmax(like_out_put)
        finish_out_put = F.softmax(finish_out_put)
        # pred = torch.sigmoid(model(Xi, Xv, video_feature, audio_feature, title_feature, title_value)).cpu()
        return like_out_put.data.numpy(), finish_out_put.data.numpy()

    def evaluate(self, Xi, Xv, video_feature, audio_feature, title_feature, title_value, like_y, finish_y):
        """
        :param Xi: tensor of feature index
        :param Xv: tensor of feature value
        :param y: tensor of labels
        :return: metric of the evaluation
        """
        like_out_put, finish_out_put = self.inner_predict_proba(Xi, Xv, video_feature, audio_feature, title_feature, title_value)
        return self.eval_metric(like_y.cpu().data.numpy(), like_out_put[:, 1]), self.eval_metric(finish_y.cpu().data.numpy(), finish_out_put[:, 1])