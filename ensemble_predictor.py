#!/usr/bin/env python
# encoding: utf-8


from keras.models import Model
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.layers import GRU, Embedding, Input, RepeatVector, Dense, Flatten
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
    tX = None
    xX = None
    Y1 = None
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
        if Y1 is None:
            Y1 = data[:, t + 1 + T]
        else:
            Y1 = np.concatenate([Y1, data[:, t + 1 + T]])

    return tX, xX, Y1


t_input = Input(shape=(1,))
x_input = Input(shape=(T,))
temb = Flatten()(Embedding(96 - T, embedding_dim_time)(t_input))
xemb = Embedding(num_locs, embedding_dim_loc, input_length=T)(x_input)
rep_time = RepeatVector(T)(temb)
merge_input = concatenate([rep_time, xemb], axis=-1)
gru1 = GRU(hidden_dim, return_sequences=True, unroll=True, activation='softsign', dropout=0.2, recurrent_dropout=0.2)(merge_input)
gru2 = GRU(hidden_dim, return_sequences=False, unroll=True, activation='softsign', dropout=0.2, recurrent_dropout=0.2)(gru1)
y = Dense(num_locs, activation='softmax')(gru2)

ensemble_predictor = Model([t_input, x_input], y)
ensemble_predictor.summary()
ensemble_predictor.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(lr=1e-3))
init_weights = ensemble_predictor.get_weights()

for d in xrange(1, 32):
    callbacks = [
        CSVLogger('../results/sadHybridHumanPredictor/ensemble_predictor_2012_jan/log_d{}.csv'.format(d), separator=',', append=False),
        ModelCheckpoint(filepath='../results/sadHybridHumanPredictor/ensemble_predictor_2012_jan/ensemble_predictor_{}.hdf5'.format(d), verbose=1, save_best_only=True, monitor='loss'),
        EarlyStopping(monitor='loss', patience=0, verbose=1, mode='auto')
    ]
    tX, xX, Y1 = read_trainingset('/home/hpc/work/data/dis_forensemble_2012_jan/', d)
    ensemble_predictor.set_weights(init_weights)
    ensemble_predictor.fit([tX, xX], Y1, batch_size=batch_size, epochs=20, shuffle=True,\
                            verbose=1, callbacks=callbacks)
