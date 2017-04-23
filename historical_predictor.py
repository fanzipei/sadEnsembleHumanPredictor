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
hidden_dim = 256
num_users = 100
num_locs = 1441
batch_size = 256
T = 4


def get_userindex(folderpath, min_day, max_day):
    user_set = set()
    user_index = dict({})
    for d in xrange(min_day, max_day + 1):
        filepath = '{}day_{}.csv'.format(folderpath, d)
        data = np.genfromtxt(filepath, dtype=np.int32, delimiter=',')
        user_set = user_set.union(set(data[:, 0]))

    for uid in user_set:
        user_index[uid] = len(user_index)

    return user_index


def read_trainingset(folderpath, min_day, max_day, user_index, max_iter):
    for i in xrange(max_iter):
        d = np.random.randint(min_day, max_day + 1)
        uX = None
        tX = None
        xX = None
        Y = None
        filepath = '{}day_{}.csv'.format(folderpath, d)
        data = np.genfromtxt(filepath, dtype=np.int32, delimiter=',')
        if uX is None:
            uX = np.vectorize(lambda x:user_index[x])(data[:, 0]).repeat(96 - T)
        else:
            uX = np.concatenate([uX, np.vectorize(lambda x:user_index[x])(data[:, 0]).repeat(96 - T - 2)])
        for t in xrange(96 - T):
            if tX is None:
                tX = np.array([t] * data.shape[0])
            else:
                tX = np.concatenate([tX, np.array([t] * data.shape[0])])
            if xX is None:
                xX = data[:, t + 1:t + 1 + T]
            else:
                xX = np.concatenate([xX, data[:, t + 1:t + 1 + T]])
            if Y is None:
                Y = data[:, t + 1 + T]
            else:
                Y = np.concatenate([Y, data[:, t + 1 + T]])

        yield uX, tX, xX, Y


user_index = get_userindex('/home/fan/work/data/dis_forhybrid/', 1, 30)
num_users = len(user_index)


u_input = Input(shape=(1,))
t_input = Input(shape=(1,))
x_input = Input(shape=(T,))
uemb = Flatten()(Embedding(num_users + 1, embedding_dim_user)(u_input))
temb = Flatten()(Embedding(96, embedding_dim_time)(t_input))
xemb = Embedding(num_locs, embedding_dim_loc, input_length=T)(x_input)
rep_uidx = RepeatVector(T)(uemb)
rep_time = RepeatVector(T)(temb)
merge_input = concatenate([rep_uidx, rep_time, xemb], axis=-1)
gru = GRU(hidden_dim, return_sequences=False, unroll=True, activation='softsign')(merge_input)
y = Dense(num_locs, activation='softmax')(gru)

historical_predictor = Model([u_input, t_input, x_input], y)
historical_predictor.summary()
historical_predictor.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(lr=1e-3))

for uX, tX, xX, Y in read_trainingset('/home/fan/work/data/dis_forhybrid/', 1, 30, user_index, 500):
    historical_predictor.fit([uX, tX, xX], Y, batch_size=batch_size, epochs=1)
    historical_predictor.save('./historical_predictor.hdf5')
    historical_predictor.save_weights('./historical_predictor_weights.hdf5')
