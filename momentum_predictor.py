#!/usr/bin/env python
# encoding: utf-8


from keras.models import Model
from keras.layers import GRU, Embedding, Input, RepeatVector, Dense, Flatten
from keras.layers.merge import concatenate
from keras.optimizers import RMSprop

import numpy as np

embedding_dim_time = 8
embedding_dim_loc = 64
hidden_dim = 64
num_locs = 1441
batch_size = 256
T = 4


def read_trainingset(folderpath, min_day, max_day, max_iter):
    for i in xrange(max_iter):
        d = np.random.randint(min_day, max_day + 1)
        tX = None
        xX = None
        Y = None
        filepath = '{}day_{}.csv'.format(folderpath, d)
        data = np.genfromtxt(filepath, dtype=np.int32, delimiter=',')
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

        yield tX, xX, Y


t_input = Input(shape=(1,))
x_input = Input(shape=(T,))
temb = Flatten()(Embedding(96, embedding_dim_time)(t_input))
xemb = Embedding(num_locs, embedding_dim_loc, input_length=T)(x_input)
rep_time = RepeatVector(T)(temb)
merge_input = concatenate([rep_time, xemb], axis=-1)
gru1 = GRU(hidden_dim, return_sequences=True, unroll=True, activation='softsign')(merge_input)
gru2 = GRU(hidden_dim, return_sequences=False, unroll=True, activation='softsign')(gru1)
y = Dense(num_locs, activation='softmax')(gru2)

momentum_predictor = Model([t_input, x_input], y)
momentum_predictor.summary()
momentum_predictor.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(lr=1e-3))

for tX, xX, Y in read_trainingset('/media/fan/HDPC-UT/ZDC/TrainingForMapping/dis_forhybrid/', 1, 30, 500):
    momentum_predictor.fit([tX, xX], Y, batch_size=batch_size, epochs=1)
    momentum_predictor.save('./momentum_predictor.hdf5')
    momentum_predictor.save_weights('./momentum_predictor_weights.hdf5')
