#!/usr/bin/env python
# encoding: utf-8


from keras.models import Model, load_model
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.layers import GRU, Embedding, Input, RepeatVector, Dense, Flatten
from keras.layers.merge import concatenate
from keras.optimizers import RMSprop

import numpy as np

embedding_dim_time = 8
embedding_dim_loc = 64
hidden_dim = 128
# tokyo
# num_locs = 1441
# osaka
num_locs = 1537
batch_size = 256
T = 4


def read_trainingset(folderpath, d):
    tX = np.zeros([0], dtype=np.int32)
    xX = np.zeros([0, T], dtype=np.int32)
    Y1 = np.zeros([0], dtype=np.int32)
    filepath = '{}day_{}.csv'.format(folderpath, d)
    data = np.genfromtxt(filepath, dtype=np.int32, delimiter=',')
    for t in xrange(96 - T):
        tX = np.concatenate([tX, np.array([t] * data.shape[0])])
        xX = np.concatenate([xX, data[:, t + 1:t + 1 + T]])
        Y1 = np.concatenate([Y1, data[:, t + 1 + T]])

    return tX, xX, Y1


# t_input = Input(shape=(1,))
# x_input = Input(shape=(T,))
# w_input = Input(shape=(1,))
# temb = Flatten()(Embedding(96 - T, embedding_dim_time)(t_input))
# xemb = Embedding(num_locs, embedding_dim_loc, input_length=T)(x_input)
# rep_time = RepeatVector(T)(temb)
# rep_w = RepeatVector(T)(w_input)
# merge_input = concatenate([rep_time, xemb, rep_w], axis=-1)
# gru1 = GRU(hidden_dim, return_sequences=True, unroll=True, activation='softsign', dropout=0.2, recurrent_dropout=0.2)(merge_input)
# gru2 = GRU(hidden_dim, return_sequences=False, unroll=True, activation='softsign', dropout=0.2, recurrent_dropout=0.2)(gru1)
# y = Dense(num_locs, activation='softmax')(gru2)

# ensemble_predictor = Model([t_input, x_input, w_input], y)
# ensemble_predictor.summary()
# ensemble_predictor.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(lr=1e-3))
# init_weights = ensemble_predictor.get_weights()

ensemble_predictor = load_model('../results/sadHybridHumanPredictor/one_predictor_2010_aug_osaka/')

weekday_jan_2011 = [1, 2, 3, 8, 9, 10, 15, 16, 22, 23, 29, 30]
weekday_may_2011 = [1, 3, 4, 5, 7, 8, 14, 15, 21, 22, 28, 29]
weekday_aug_2010 = [1, 7, 8, 14, 15, 21, 22, 28, 29]
weekday_oct_2010 = [2, 3, 9, 10, 11, 16, 17, 23, 24, 30, 31]
weekday_jan_2012 = [1, 2, 3, 7, 8, 9, 14, 15, 21, 22, 28, 29]
weekday_may_2012 = [3, 4, 5, 6, 12, 13, 19, 20, 26, 27]
weekday_aug_2012 = [4, 5, 11, 12, 18, 19, 25, 26]
weekday_oct_2012 = [6, 7, 8, 13, 14, 20, 21, 27, 28]

open('../results/sadHybridHumanPredictor/one_predictor_2012_aug_osaka.csv', 'w').close()

for d in xrange(1, 32):
    tX, xX, Y1 = read_trainingset('/home/fan/work/data/dis_forensemble_2012_aug_osaka/', d)
    w = None

    if d in weekday_aug_2012:
        w = np.concatenate([w, 1 + np.zeros([tX.shape[0], 1])])
    else:
        w = np.concatenate([w, -1 + np.zeros([tX.shape[0], 1])])

    loss = ensemble_predictor.evaluate([tX, xX, w], Y1, batch_size=4096)
    with open('../results/sadHybridHumanPredictor/one_predictor_2012_aug_osaka.csv', 'a') as f:
        for r in loss:
            f.write('{},'.format(r))
        f.write('\n')
