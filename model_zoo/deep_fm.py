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

import torch.backends.cudnn

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

    def __init__(self, field_size, tile_word_size, feature_sizes, video_feature_size, target, embedding_size=32, is_shallow_dropout=True, dropout_shallow=[0.5, 0.5],
                 h_depth=2, deep_layers=[32, 32], is_deep_dropout=True, dropout_deep=[0.5, 0.5, 0.5],
                 deep_layers_activation='relu', n_epochs=64, batch_size=256, learning_rate=0.003,
                 optimizer_type='adam', is_batch_norm=False, verbose=False, random_seed=950104, weight_decay=0.0,
                 use_fm=True, use_ffm=False, use_deep=True, loss_type='logloss', eval_metric=roc_auc_score,
                 use_cuda=True, n_class=1, greater_is_better=True
                 ):
        super(DeepFM, self).__init__()
        self.total_count = 0
        self.target = target
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

        torch.manual_seed(self.random_seed)

        """
            check cuda
        """
        if self.use_cuda and not torch.cuda.is_available():
            self.use_cuda = False
            print("Cuda is not available, automatically changed into cpu model")

        """
            check use fm or ffm
        """
        if self.use_fm and self.use_ffm:
            print("only support one type only, please make sure to choose only fm or ffm part")
            exit(1)
        elif self.use_fm and self.use_deep:
            print("The model is deepfm(fm+deep layers)")
        elif self.use_ffm and self.use_deep:
            print("The model is deepffm(ffm+deep layers)")
        elif self.use_fm:
            print("The model is fm only")
        elif self.use_ffm:
            print("The model is ffm only")
        elif self.use_deep:
            print("The model is deep layers only")
        else:
            print("You have to choose more than one of (fm, ffm, deep) models to use")
            exit(1)

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
        if self.use_fm:
            print("Init fm part")
            self.fm_first_order_embeddings = nn.ModuleList(
                [nn.Embedding(feature_size, 1) for feature_size in self.feature_sizes])
            if self.dropout_shallow:
                self.fm_first_order_dropout = nn.Dropout(self.dropout_shallow[0])
            self.fm_second_order_embeddings = nn.ModuleList(
                [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes])
            if self.dropout_shallow:
                self.fm_second_order_dropout = nn.Dropout(self.dropout_shallow[1])
            print("Init fm part succeed")

        """
            ffm part
        """
        if self.use_ffm:
            print("Init ffm part")
            self.ffm_first_order_embeddings = nn.ModuleList(
                [nn.Embedding(feature_size, 1) for feature_size in self.feature_sizes])
            if self.dropout_shallow:
                self.ffm_first_order_dropout = nn.Dropout(self.dropout_shallow[0])
            self.ffm_second_order_embeddings = nn.ModuleList(
                [nn.ModuleList([nn.Embedding(feature_size, self.embedding_size) for i in range(self.field_size)]) for
                 feature_size in self.feature_sizes])
            if self.dropout_shallow:
                self.ffm_second_order_dropout = nn.Dropout(self.dropout_shallow[1])
            print("Init ffm part succeed")

        """
            deep part
        """
        if self.use_deep:
            print("Init deep part")
            if not self.use_fm and not self.use_ffm:
                self.fm_second_order_embeddings = nn.ModuleList(
                    [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes])

            if self.is_deep_dropout:
                self.linear_0_dropout = nn.Dropout(self.dropout_deep[0])

            self.linear_1 = nn.Linear((self.field_size+2) * self.embedding_size, deep_layers[0])
            if self.is_batch_norm:
                self.batch_norm_1 = nn.BatchNorm1d(deep_layers[0])
            if self.is_deep_dropout:
                self.linear_1_dropout = nn.Dropout(self.dropout_deep[1])
            for i, h in enumerate(self.deep_layers[1:], 1):
                setattr(self, 'linear_' + str(i + 1), nn.Linear(self.deep_layers[i - 1], self.deep_layers[i]))
                if self.is_batch_norm:
                    setattr(self, 'batch_norm_' + str(i + 1), nn.BatchNorm1d(deep_layers[i]))
                if self.is_deep_dropout:
                    setattr(self, 'linear_' + str(i + 1) + '_dropout', nn.Dropout(self.dropout_deep[i + 1]))

            print("Init deep part succeed")

        print("Init succeed")

    def forward(self, Xi, Xv, video_feature, title_feature, title_value):
        """
        :param Xi_train: index input tensor, batch_size * k * 1
        :param Xv_train: value input tensor, batch_size * k * 1
        :return: the last output
        """
        """
            fm part
        """
        if self.use_fm:
            # print("test", Xi[:, 0, :])
            # exit()

            fm_first_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in
                                      enumerate(self.fm_first_order_embeddings)]
            # fm_first_order_emb_arr = list()
            # for i, emb in enumerate(self.fm_first_order_embeddings):
            #     print(Xi[:, i, :], emb)
            #     # print(Xi[:, i, :], emb, torch.sum(emb(Xi[:, i, :]), 1).t(), Xv[:, i])
            #     fm_first_order_emb_arr.append((torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t())

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
            if self.is_shallow_dropout:
                fm_second_order = self.fm_second_order_dropout(fm_second_order)

        """
            ffm part
        """
        if self.use_ffm:
            ffm_first_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in
                                       enumerate(self.ffm_first_order_embeddings)]
            ffm_first_order = torch.cat(ffm_first_order_emb_arr, 1)
            if self.is_shallow_dropout:
                ffm_first_order = self.ffm_first_order_dropout(ffm_first_order)
            ffm_second_order_emb_arr = [[(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for emb in f_embs] for
                                        i, f_embs in enumerate(self.ffm_second_order_embeddings)]
            ffm_wij_arr = []
            for i in range(self.field_size):
                for j in range(i + 1, self.field_size):
                    ffm_wij_arr.append(ffm_second_order_emb_arr[i][j] * ffm_second_order_emb_arr[j][i])
            ffm_second_order = sum(ffm_wij_arr)
            if self.is_shallow_dropout:
                ffm_second_order = self.ffm_second_order_dropout(ffm_second_order)

        """
            deep part
        """
        if self.use_deep:

            if self.use_fm:
                deep_emb = torch.cat(fm_second_order_emb_arr, 1)
            elif self.use_ffm:
                deep_emb = torch.cat([sum(ffm_second_order_embs) for ffm_second_order_embs in ffm_second_order_emb_arr],
                                     1)
            else:
                deep_emb = torch.cat([(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in
                                      enumerate(self.fm_second_order_embeddings)], 1)

            if video_feature is not None:
                video_feature = self.video_line(video_feature)
                deep_emb = torch.cat([deep_emb, video_feature], 1)

            title_embedding = self.title_embedding(title_feature)
            title_embedding = title_embedding*title_value
            title_embedding = title_embedding.permute(0, 2, 1)
            title_embedding = torch.sum(title_embedding, -1)
            deep_emb = torch.cat([deep_emb, title_embedding], 1)
            # print(deep_emb.shape)

            if self.deep_layers_activation == 'sigmoid':
                activation = torch.sigmoid
            elif self.deep_layers_activation == 'tanh':
                activation = F.tanh
            else:
                activation = F.relu
            if self.is_deep_dropout:
                deep_emb = self.linear_0_dropout(deep_emb)
            x_deep = self.linear_1(deep_emb)
            if self.is_batch_norm:
                x_deep = self.batch_norm_1(x_deep)
            x_deep = activation(x_deep)
            if self.is_deep_dropout:
                x_deep = self.linear_1_dropout(x_deep)
            for i in range(1, len(self.deep_layers)):
                x_deep = getattr(self, 'linear_' + str(i + 1))(x_deep)
                if self.is_batch_norm:
                    x_deep = getattr(self, 'batch_norm_' + str(i + 1))(x_deep)
                x_deep = activation(x_deep)
                if self.is_deep_dropout:
                    x_deep = getattr(self, 'linear_' + str(i + 1) + '_dropout')(x_deep)
        """
            sum
        """
        if self.use_fm and self.use_deep:
            total_sum = torch.sum(fm_first_order, 1) + torch.sum(fm_second_order, 1) + torch.sum(x_deep, 1) + self.bias
        elif self.use_ffm and self.use_deep:
            total_sum = torch.sum(ffm_first_order, 1) + torch.sum(ffm_second_order, 1) + torch.sum(x_deep, 1) + self.bias
        elif self.use_fm:
            total_sum = torch.sum(fm_first_order, 1) + torch.sum(fm_second_order, 1) + self.bias
        elif self.use_ffm:
            total_sum = torch.sum(ffm_first_order, 1) + torch.sum(ffm_second_order, 1) + self.bias
        else:
            total_sum = torch.sum(x_deep, 1)
        return total_sum

    def fit2(self, model, optimizer, criterion, Xi_train, Xv_train, video_feature, title_feature, title_value,
             y_like_train, y_finish_train, count, Xi_valid=None,
             Xv_valid=None, y_like_valid=None, y_finish_valid=None, video_feature_val=None, title_feature_val=None,
             title_value_val=None, ealry_stopping=False, save_path=None):

        # if save_path and not os.path.exists('/'.join(save_path.split('/')[0:-1])):
        #     print("Save path is not existed!")
        #     return

        if self.verbose:
            print("pre_process data ing...")
        is_valid = False

        Xi_train = np.array(Xi_train).reshape((-1, self.field_size, 1))
        # video_feature = np.array(video_feature)
        title_feature = np.array(title_feature)
        title_value = [[[j for _ in range(self.embedding_size)] for j in i] for i in title_value]
        title_value = np.array(title_value)
        Xv_train = np.array(Xv_train)
        y_like_train = np.array(y_like_train)
        y_finish_train = np.array(y_finish_train)
        if self.target == "finish":
            y_train = y_finish_train
        elif self.target == "like":
            y_train = y_like_train
        else:
            print("target wrong")
            return
        x_size = Xi_train.shape[0]
        if Xi_valid:
            Xi_valid = np.array(Xi_valid).reshape((-1, self.field_size, 1))
            Xv_valid = np.array(Xv_valid)

            title_feature_val = np.array(title_feature_val)
            title_value_val = [[[j for _ in range(self.embedding_size)] for j in i] for i in title_value_val]
            title_value_val = np.array(title_value_val)

            y_like_valid = np.array(y_like_valid)
            y_finish_valid = np.array(y_finish_valid)
            if self.target == "finish":
                y_valid = y_finish_valid
            elif self.target == "like":
                y_valid = y_like_valid
            else:
                print("target wrong")
                return
            x_valid_size = Xi_valid.shape[0]
            is_valid = True
        if self.verbose:
            print("pre_process data finished")

        train_result = []
        valid_result = []
        total_loss = 0.0
        batch_iter = x_size // self.batch_size
        epoch_begin_time = time()
        batch_begin_time = time()
        for i in range(batch_iter + 1):
            self.total_count += 1
            offset = i * self.batch_size
            end = min(x_size, offset + self.batch_size)
            if offset == end:
                break
            batch_xi = Variable(torch.LongTensor(Xi_train[offset:end]))
            batch_xv = Variable(torch.FloatTensor(Xv_train[offset:end]))
            batch_like_y = Variable(torch.FloatTensor(y_like_train[offset:end]))
            batch_finish_y = Variable(torch.FloatTensor(y_finish_train[offset:end]))

            try:
                batch_video_feature = Variable(torch.FloatTensor(video_feature[offset:end]))
            except:
                print("fucked", video_feature[offset:end])
                continue
            batch_title_value = Variable(torch.FloatTensor(title_value[offset:end]))
            batch_title_feature = Variable(torch.LongTensor(title_feature[offset:end]))

            if self.use_cuda:
                batch_xi, batch_xv, batch_like_y, batch_finish_y, batch_video_feature, batch_title_value, batch_title_feature = \
                    batch_xi.cuda(), batch_xv.cuda(), batch_like_y.cuda(), batch_finish_y.cuda(), batch_video_feature.cuda(), \
                    batch_title_value.cuda(), batch_title_feature.cuda()
            optimizer.zero_grad()

            outputs = model(batch_xi, batch_xv, batch_video_feature, batch_title_feature, batch_title_value)
            if self.target == "finish":
                batch_y = batch_finish_y
            elif self.target == "like":
                batch_y = batch_like_y
            else:
                print("target wrong")
                return
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.data
            if self.verbose:
                if i % 100 == 99:  # print every 100 mini-batches
                    eval = self.evaluate(batch_xi, batch_xv, batch_video_feature, batch_title_feature, batch_title_value, batch_y)
                    print('[%d, %5d] loss: %.6f metric: %.6f time: %.1f s' %
                          (count + 1, i + 1, total_loss / 100.0, eval, time() - batch_begin_time))
                    total_loss = 0.0
                    batch_begin_time = time()
            if self.total_count % 100 == 0:
                print("total count", self.total_count)
            if save_path and self.total_count % 5000 == 0:
                torch.save(self.state_dict(), os.path.join(save_path, "byte_%s.model" % self.total_count))

        train_loss, train_eval = self.eval_by_batch(Xi_train, Xv_train, y_train, x_size, video_feature, title_feature, title_value)
        train_result.append(train_eval)
        print('*' * 50)
        print('[%d] loss: %.6f metric: %.6f time: %.1f s' %
              (count + 1, train_loss, train_eval, time() - epoch_begin_time))
        print('*' * 50)

        if is_valid:
            valid_loss, valid_eval = self.eval_by_batch(Xi_valid, Xv_valid, y_valid, x_valid_size,
                                                        video_feature_val, title_feature_val, title_value_val)
            # valid_result.append(valid_eval)
            print('valid*' * 20)
            print('[%d] loss: %.6f metric: %.6f time: %.1f s' %
                  (count + 1, valid_loss, valid_eval, time() - epoch_begin_time))
            print('valid*' * 20)

    def eval_by_batch(self, Xi, Xv, y, x_size, video_feature, title_feature, title_value):
        total_loss = 0.0
        y_pred = []
        if self.use_ffm:
            batch_size = 16384 * 2
        else:
            batch_size = 16384
        batch_iter = x_size // batch_size
        criterion = F.binary_cross_entropy_with_logits
        model = self.eval()
        for i in range(batch_iter + 1):
            offset = i * batch_size
            end = min(x_size, offset + batch_size)
            if offset == end:
                break
            batch_xi = Variable(torch.LongTensor(Xi[offset:end]))
            batch_xv = Variable(torch.FloatTensor(Xv[offset:end]))
            batch_y = Variable(torch.FloatTensor(y[offset:end]))
            batch_video_feature = Variable(torch.FloatTensor(video_feature[offset:end]))
            batch_title_feature = Variable(torch.LongTensor(title_feature[offset:end]))
            batch_title_value = Variable(torch.FloatTensor(title_value[offset:end]))
            if self.use_cuda:
                batch_xi, batch_xv, batch_y, batch_video_feature, batch_title_feature, batch_title_value = \
                    batch_xi.cuda(), batch_xv.cuda(), batch_y.cuda(), batch_video_feature.cuda(),\
                    batch_title_feature.cuda(), batch_title_value.cuda()
            outputs = model(batch_xi, batch_xv, batch_video_feature, batch_title_feature, batch_title_value)
            pred = torch.sigmoid(outputs).cpu()
            y_pred.extend(pred.data.numpy())
            loss = criterion(outputs, batch_y)
            total_loss += loss.data * (end - offset)
        total_metric = self.eval_metric(y, y_pred)
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
                if valid_result[-1] < valid_result[-2] and \
                        valid_result[-2] < valid_result[-3] and \
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

    def predict_proba(self, Xi, Xv, video_feature, title_feature, title_value):
        Xi = np.array(Xi).reshape((-1, self.field_size, 1))
        Xi = Variable(torch.LongTensor(Xi))
        Xv = Variable(torch.FloatTensor(Xv))
        video_feature = Variable(torch.FloatTensor(video_feature))
        title_value = Variable(torch.FloatTensor(title_value))
        title_feature = Variable(torch.LongTensor(title_feature))

        if self.use_cuda:
            Xi, Xv, video_feature, title_value, title_feature = \
                Xi.cuda(), Xi.cuda(), video_feature.cuda(), \
                title_value.cuda(), title_feature.cuda()

        model = self.eval()
        pred = torch.sigmoid(model(Xi, Xv, video_feature, title_feature, title_value)).cpu()
        return pred.data.numpy()

    def inner_predict(self, Xi, Xv):
        """
        :param Xi: tensor of feature index
        :param Xv: tensor of feature value
        :return: output, numpy
        """
        model = self.eval()
        pred = torch.sigmoid(model(Xi, Xv)).cpu()
        return (pred.data.numpy() > 0.5)

    def inner_predict_proba(self, Xi, Xv, video_feature, title_feature, title_value):
        """
        :param Xi: tensor of feature index
        :param Xv: tensor of feature value
        :return: output, numpy
        """
        model = self.eval()
        pred = torch.sigmoid(model(Xi, Xv, video_feature, title_feature, title_value)).cpu()
        return pred.data.numpy()

    def evaluate(self, Xi, Xv, video_feature, title_feature, title_value, y):
        """
        :param Xi: tensor of feature index
        :param Xv: tensor of feature value
        :param y: tensor of labels
        :return: metric of the evaluation
        """
        y_pred = self.inner_predict_proba(Xi, Xv, video_feature, title_feature, title_value)
        return self.eval_metric(y.cpu().data.numpy(), y_pred)
