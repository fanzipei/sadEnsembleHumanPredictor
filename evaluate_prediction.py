#!/usr/bin/env python
# encoding: utf-8

from keras.layers.merge import dot, concatenate
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.layers import GRU, Embedding, Input, RepeatVector, Dense, Flatten, Reshape
from keras.models import Model, load_model
from keras.optimizers import RMSprop
import numpy as np


embedding_dim_time = 8
embedding_dim_loc = 64
num_models = 31
hidden_dim = 128
num_locs = 1441
batch_size = 4096
T = 4

def read_trainingset(folderpath, d):
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


models = [build_and_load_model('../results/sadHybridHumanPredictor/ensemble_predictor_2010_aug/ensemble_predictor_{}.hdf5'.format(i)) for i in xrange(1, num_models + 1)]
open('../results/sadHybridHumanPredictor/ensemble_losslog_2012_aug.csv', 'w').close()
t_input = Input(shape=(1,))
x_input = Input(shape=(T,))
xemb = Embedding(num_locs, embedding_dim_loc, input_length=T)(x_input)
gru1 = GRU(hidden_dim, return_sequences=False, unroll=True, activation='softsign')(xemb)
models_out = [[Reshape((num_locs, 1))(model([t_input, x_input])[i]) for model in models] for i in xrange(4)]
preds = [concatenate(models_out[i], axis=-1) for i in xrange(4)]
weights = Reshape((num_models, 1))(Dense(num_models, activation='softmax')(gru1))
y1 = Reshape((num_locs,))(dot([preds[0], weights], [2, 1]))
y2 = Reshape((num_locs,))(dot([preds[1], weights], [2, 1]))
y3 = Reshape((num_locs,))(dot([preds[2], weights], [2, 1]))
y4 = Reshape((num_locs,))(dot([preds[3], weights], [2, 1]))
online_predictor = Model([t_input, x_input], [y1, y2, y3, y4])
online_predictor.summary()
online_predictor.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(lr=1e-3))

for d in xrange(1, 32):
    X = read_trainingset('/home/fan/work/data/dis_forensemble_2012_aug/', d)
    for t in xrange(96 - 3 * T + 1):
        print 'Day {}, time {}'.format(d, t)
        online_predictor.load_weights('/home/fan/work/results/sadHybridHumanPredictor/online_predictor_2012_aug/online_predictor_d{}t{}.hdf5'.format(d, t))
        tX, xX, Y1, Y2, Y3, Y4 = X[t]
        loss = online_predictor.evaluate([tX, xX], [Y1, Y2, Y3, Y4], batch_size=batch_size)
        print loss
        for i in xrange(num_models):
            loss += models[i].evaluate([tX, xX], [Y1, Y2, Y3, Y4], batch_size=batch_size)
        with open('../results/sadHybridHumanPredictor/ensemble_losslog_2012_aug.csv', 'a') as f:
            for r in loss:
                f.write('{},'.format(r))
            f.write('\n')

