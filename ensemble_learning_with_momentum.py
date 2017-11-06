#!/usr/bin/env python
# encoding: utf-8

from keras import backend as K
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
# tokyo
# num_locs = 1441
# osaka
num_locs = 1537
batch_size = 256
T = 4

def read_trainingset(folderpath, d):
    tX = None
    xX = None
    Y1 = None
    filepath = '{}day_{}.csv'.format(folderpath, d)
    data = np.genfromtxt(filepath, dtype=np.int32, delimiter=',')
    X = []
    for t in xrange(96 - T):
        tX = np.array([t] * data.shape[0])
        xX = data[:, t + 1:t + 1 + T]
        Y1 = data[:, t + 1 + T]
        X.append((tX, xX, Y1))

    return X


def build_and_load_model(model_path):
    print 'Build model {}'.format(model_path)
    t_input = Input(shape=(1,))
    x_input = Input(shape=(T,))
    temb = Flatten()(Embedding(96 - T, embedding_dim_time)(t_input))
    xemb = Embedding(num_locs, embedding_dim_loc, input_length=T)(x_input)
    rep_time = RepeatVector(T)(temb)
    merge_input = concatenate([rep_time, xemb], axis=-1)
    gru1 = GRU(hidden_dim, return_sequences=True, unroll=True, activation='softsign', dropout=0.2, recurrent_dropout=0.2)(merge_input)
    gru2 = GRU(hidden_dim, return_sequences=False, unroll=True, activation='softsign', dropout=0.2, recurrent_dropout=0.2)(gru1)
    y = Dense(num_locs, activation='softmax')(gru2)

    predictor = Model([t_input, x_input], y)
    predictor.trainable = False
    predictor.load_weights(model_path)

    return predictor


models = [build_and_load_model('../results/sadHybridHumanPredictor/ensemble_predictor_2011_jan_osaka/ensemble_predictor_{}.hdf5'.format(i)) for i in xrange(1, num_models + 1)]
print 'Load Models Finished'

t_input = Input(shape=(1,))
x_input = Input(shape=(T,))
xemb = Embedding(num_locs, embedding_dim_loc, input_length=T)(x_input)
gru1 = GRU(hidden_dim, return_sequences=False, unroll=True, activation='softsign')(xemb)
xemb_momentum = Embedding(num_locs, embedding_dim_loc, input_length=T)(x_input)
gru_momentum_1 = GRU(hidden_dim, return_sequences=True, unroll=True, activation='softsign', dropout=0.2, recurrent_dropout=0.2)(xemb_momentum)
gru_momentum_2 = GRU(hidden_dim, return_sequences=False, unroll=True, activation='softsign', dropout=0.2, recurrent_dropout=0.2)(gru_momentum_1)
out_momentum = Dense(num_locs, activation='softmax')(gru_momentum_2)

momentum_predictor = Model(x_input, out_momentum)
momentum_predictor.summary()

models_out = [Reshape((num_locs, 1))(model([t_input, x_input])) for model in models]
models_out.append(Reshape((num_locs, 1))(momentum_predictor(x_input)))
pred = concatenate(models_out, axis=-1)
weights = Reshape((num_models + 1, 1))(Dense(num_models + 1, activation='softmax')(gru1))
y = Reshape((num_locs,))(dot([pred, weights], [2, 1]))


online_predictor = Model([t_input, x_input], y)
online_predictor.summary()
momentum_predictor.trainable = False
online_predictor.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(lr=1e-3))

momentum_predictor.trainable = True
momentum_predictor.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(lr=1e-3))

open('../results/sadHybridHumanPredictor/ensemble_predictor_2012_may_osaka_corrected.csv', 'w').close()

for d in xrange(1, 32):
    X = read_trainingset('/home/fan/work/data/dis_forensemble_2012_jan_osaka/', d)
    for t in xrange(96 - T - 1):
        callbacks = [
            ModelCheckpoint(filepath='../results/sadHybridHumanPredictor/online_predictor_with_momentum_2012_jan_osaka/online_predictor_d{}t{}.hdf5'.format(d, t),\
                            verbose=1, monitor='loss', save_best_only=True),
            EarlyStopping(monitor='loss', patience=0, verbose=1, mode='auto')
        ]
        tX, xX, Y1 = X[t]
        online_predictor.fit([tX, xX], Y1, batch_size=batch_size, epochs=5, shuffle=True, verbose=1, callbacks=callbacks)
        momentum_predictor.fit(xX, Y1, batch_size=batch_size, epochs=25, shuffle=True, verbose=1, callbacks=[EarlyStopping(monitor='loss', patience=0, verbose=1, mode='auto')])

        tX, xX, Y1 = X[t + 1]
        loss = online_predictor.evaluate([tX, xX], Y1, batch_size=4096)
        with open('../results/sadHybridHumanPredictor/ensemble_predictor_2012_may_osaka_corrected.csv', 'a') as f:
            f.write('{}\n'.format(loss))
