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
    filepath = '{}day_{}.csv'.format(folderpath, d)
    data = np.genfromtxt(filepath, dtype=np.int32, delimiter=',')
    X = []
    for t in xrange(96 - T - 3):
        tX = np.array([t] * data.shape[0])
        xX = data[:, t + 1:t + 1 + T]
        Y1 = data[:, t + 1 + T]
        Y2 = data[:, t + 2 + T]
        Y3 = data[:, t + 3 + T]
        Y4 = data[:, t + 4 + T]
        X.append((tX, xX, Y1, Y2, Y3, Y4))

    return X


x_input = Input(shape=(T,))
xemb = Embedding(num_locs, embedding_dim_loc, input_length=T)(x_input)
gru1 = GRU(hidden_dim, return_sequences=True, unroll=True, activation='softsign')(xemb)
gru21 = GRU(hidden_dim, return_sequences=False, unroll=True, activation='softsign')(gru1)
gru22 = GRU(hidden_dim, return_sequences=False, unroll=True, activation='softsign')(gru1)
gru23 = GRU(hidden_dim, return_sequences=False, unroll=True, activation='softsign')(gru1)
gru24 = GRU(hidden_dim, return_sequences=False, unroll=True, activation='softsign')(gru1)
shared_softmax = Dense(num_locs, activation='softmax')
y1 = shared_softmax(gru21)
y2 = shared_softmax(gru22)
y3 = shared_softmax(gru23)
y4 = shared_softmax(gru24)

momentum_predictor = Model(x_input, [y1, y2, y3, y4])
momentum_predictor.summary()
momentum_predictor.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(lr=5e-3))

for d in xrange(1, 32):
    X = read_trainingset('/home/fan/work/data/dis_forensemble_2012_aug/', d)
    for t in xrange(96 - 3 * T + 1):
        callbacks = [
            ModelCheckpoint(filepath='../results/sadHybridHumanPredictor/momentum_predictor_2012_aug/momentum_predictor_d{}t{}.hdf5'.format(d, t),\
                            verbose=1, monitor='loss', save_best_only=True),
            EarlyStopping(monitor='loss', patience=0, verbose=1, mode='auto')
        ]
        tX, xX, Y1, Y2, Y3, Y4 = X[t]
        momentum_predictor.fit(xX, [Y1, Y2, Y3, Y4], batch_size=batch_size, epochs=100, shuffle=True, verbose=1, callbacks=callbacks)
        tX, xX, Y1, Y2, Y3, Y4 = X[t + T]
        loss = momentum_predictor.evaluate(xX, [Y1, Y2, Y3, Y4], batch_size=8192)
        with open('../results/sadHybridHumanPredictor/momentum_losslog_2012_aug.csv', 'a') as f:
            for r in loss:
                f.write('{},'.format(r))
            f.write('\n')
