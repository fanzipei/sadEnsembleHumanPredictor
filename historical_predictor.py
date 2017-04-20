#!/usr/bin/env python
# encoding: utf-8

from keras.models import Model
from keras.layers import GRU, Embedding, Input, RepeatVector, Dense, Flatten
from keras.layers.merge import concatenate
from keras.optimizers import RMSprop

import numpy as np

embedding_dim_user = 32
embedding_dim_time = 8
embedding_dim_loc = 64
hidden_dim = 64
num_users = 100
num_locs = 1441
T = 4


# def read_trainingset(folderpath, min_day, max_day):
    # for d in xrange(min_day, max_day + 1):
        # filepath = '{}day_{}.csv'.format(d)
        # data = np.genfromtxt(filepath, dtype=np.int32, delimiter=',')
        # for t in xrange(96 - T - 2):
            # data[:, t + 1:t + 1 + T]


u_input = Input(shape=(1,))
t_input = Input(shape=(1,))
x_input = Input(shape=(T,))
uemb = Flatten()(Embedding(num_users, embedding_dim_user)(u_input))
Model(u_input, uemb).summary()
temb = Flatten()(Embedding(96, embedding_dim_time)(t_input))
xemb = Embedding(num_locs, embedding_dim_loc, input_length=T)(x_input)
rep_uidx = RepeatVector(T)(uemb)
Model(u_input, rep_uidx).summary()
rep_time = RepeatVector(T)(temb)
# merge_input = merge((rep_uidx, rep_time, xemb), mode='concat', concat_axis=2)
merge_input = concatenate([rep_uidx, rep_time, xemb], axis=-1)
gru1 = GRU(hidden_dim, return_sequences=True, unroll=True, activation='softsign')(merge_input)
gru2 = GRU(hidden_dim, return_sequences=False, unroll=True, activation='softsign')(gru1)
y = Dense(num_locs, activation='softmax')(gru2)

historical_predictor = Model([u_input, t_input, x_input], y)
historical_predictor.summary()
historical_predictor.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(lr=1e-3))
