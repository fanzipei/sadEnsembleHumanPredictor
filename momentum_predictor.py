#!/usr/bin/env python
# encoding: utf-8


from keras.models import Model
from keras.layers import GRU, Embedding, Input, RepeatVector, Dense, Flatten
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.layers.merge import concatenate
from keras.optimizers import RMSprop

import numpy as np

embedding_dim_time = 8
embedding_dim_loc = 64
hidden_dim = 128
num_locs = 1441
batch_size = 256
T = 4


def read_trainingset(folderpath, d):
    xX = np.zeros([0, T], dtype=np.int32)
    Y1 = np.zeros([0], dtype=np.int32)
    filepath = '{}day_{}.csv'.format(folderpath, d)
    data = np.genfromtxt(filepath, dtype=np.int32, delimiter=',')

    X = []
    for t in xrange(96 - T):
        xX = data[:, t + 1:t + 1 + T]
        Y1 = data[:, t + 1 + T]
        X.append((xX, Y1))

    return X


x_input = Input(shape=(T,))
xemb = Embedding(num_locs, embedding_dim_loc, input_length=T)(x_input)
gru1 = GRU(hidden_dim, return_sequences=True, unroll=True, activation='softsign', dropout=0.2, recurrent_dropout=0.2)(xemb)
gru2 = GRU(hidden_dim, return_sequences=False, unroll=True, activation='softsign', dropout=0.2, recurrent_dropout=0.2)(gru1)
y = Dense(num_locs, activation='softmax')(gru2)

momentum_predictor = Model(x_input, y)
momentum_predictor.summary()
momentum_predictor.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(lr=2e-3))
open('../results/sadHybridHumanPredictor/momentum_losslog_2012_oct_tokyo.csv', 'w').close()

for d in xrange(1, 32):
    X = read_trainingset('/home/fan/work/data/dis_forensemble_2012_oct_tokyo/', d)
    for t in xrange(96 - T - 1):
        if d == 1 and t == 0:
            max_iter = 50
        else:
            max_iter = 5
        callbacks = [
            ModelCheckpoint(filepath='../results/sadHybridHumanPredictor/momentum_predictor_2012_oct_tokyo/momentum_predictor_d{}t{}.hdf5'.format(d, t),
                            verbose=1, monitor='loss', save_best_only=True),
        ]
        xX, Y1 = X[t]
        momentum_predictor.fit(xX, Y1, batch_size=batch_size, epochs=max_iter, shuffle=True, verbose=1, callbacks=callbacks)
        xX, Y1 = X[t + 1]
        loss = momentum_predictor.evaluate(xX, Y1, batch_size=8192)
        with open('../results/sadHybridHumanPredictor/momentum_losslog_2012_oct_tokyo.csv', 'a') as f:
            for r in loss:
                f.write('{},'.format(r))
            f.write('\n')
