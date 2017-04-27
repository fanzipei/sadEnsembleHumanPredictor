#!/usr/bin/env python
# encoding: utf-8

from keras.layers.merge import concatenate
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.layers import GRU, Embedding, Input, RepeatVector, Dense, Flatten, Reshape
from keras.models import Model, load_model
from keras.optimizers import RMSprop
import numpy as np


embedding_dim_time = 8
embedding_dim_loc = 64
num_models = 27
hidden_dim = 100
num_locs = 1441
batch_size = 256
T = 4

def read_trainingset(folderpath, d):
    tX = None
    xX = None
    Y1 = None
    Y2 = None
    Y3 = None
    Y4 = None
    filepath = '{}day_{}.csv'.format(folderpath, d)
    data = np.genfromtxt(filepath, dtype=np.int32, delimiter=',')
    X = []
    for t in xrange(T, 96 - 2 * T + 1):
        tX = np.array([t] * data.shape[0])
        xX = data[:, t + 1:t + 1 + T]
        Y1 = data[:, t + 1 + T]
        Y2 = data[:, t + 2 + T]
        Y3 = data[:, t + 3 + T]
        Y4 = data[:, t + 4 + T]
        X.append((tX, xX, Y1, Y2, Y3, Y4))

    return X


def build_and_load_model(model_path):
    print 'Build model {}'.format(model_path)
    t_input = Input(shape=(1,))
    x_input = Input(shape=(T,))
    temb = Flatten()(Embedding(96 - T - 3, embedding_dim_time)(t_input))
    xemb = Embedding(num_locs, embedding_dim_loc, input_length=T)(x_input)
    rep_time = RepeatVector(T)(temb)
    merge_input = concatenate([rep_time, xemb], axis=-1)
    gru1 = GRU(hidden_dim, return_sequences=True, unroll=True, activation='softsign')(merge_input)
    gru21 = GRU(hidden_dim, return_sequences=False, unroll=True, activation='softsign')(gru1)
    gru22 = GRU(hidden_dim, return_sequences=False, unroll=True, activation='softsign')(gru1)
    gru23 = GRU(hidden_dim, return_sequences=False, unroll=True, activation='softsign')(gru1)
    gru24 = GRU(hidden_dim, return_sequences=False, unroll=True, activation='softsign')(gru1)
    shared_softmax = Dense(num_locs, activation='softmax')
    y1 = shared_softmax(gru21)
    y2 = shared_softmax(gru22)
    y3 = shared_softmax(gru23)
    y4 = shared_softmax(gru24)

    predictor = Model([t_input, x_input], [y1, y2, y3, y4])
    predictor.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(lr=1e-3))
    predictor.trainable = False
    predictor.load_weights(model_path)

    return predictor

ensemble_predictor = load_model('/home/fan/work/results/sadHybridHumanPredictor/online_predictor/online_predictor_d32t0.hdf5')
print 'Load ensemble predictor finished'
models = [build_and_load_model('../results/sadHybridHumanPredictor/ensemble_predictor_v2/ensemble_predictor_{}.hdf5'.format(i)) for i in xrange(1, num_models + 1)]
open('../results/sadHybridHumanPredictor/ensemble_losslog.csv', 'w').close()

for d in xrange(32, 62):
    X = read_trainingset('/home/fan/work/data/dis_forensemble/', d)
    for t in xrange(96 - 3 * T + 1):
        print 'Day {}, time {}'.format(d, t)
        ensemble_predictor.load_weights('/home/fan/work/results/sadHybridHumanPredictor/online_predictor/online_predictor_d{}t{}.hdf5'.format(d, t))
        ensemble_predictor.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(lr=1e-3))
        tX, xX, Y1, Y2, Y3, Y4 = X[t]
        loss = ensemble_predictor.evaluate([tX, xX], [Y1, Y2, Y3, Y4], batch_size=batch_size)
        print loss
        for i in xrange(num_models):
            loss += models[i].evaluate([tX, xX], [Y1, Y2, Y3, Y4], batch_size=batch_size)
        with open('../results/sadHybridHumanPredictor/ensemble_losslog.csv', 'a') as f:
            for r in loss:
                f.write('{},'.format(r))
            f.write('\n')

