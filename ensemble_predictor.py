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
    Y2 = None
    Y3 = None
    Y4 = None
    filepath = '{}day_{}.csv'.format(folderpath, d)
    data = np.genfromtxt(filepath, dtype=np.int32, delimiter=',')
    for t in xrange(96 - T - 3):
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
        if Y2 is None:
            Y2 = data[:, t + 2 + T]
        else:
            Y2 = np.concatenate([Y2, data[:, t + 2 + T]])
        if Y3 is None:
            Y3 = data[:, t + 3 + T]
        else:
            Y3 = np.concatenate([Y3, data[:, t + 3 + T]])
        if Y4 is None:
            Y4 = data[:, t + 4 + T]
        else:
            Y4 = np.concatenate([Y4, data[:, t + 4 + T]])

    return tX, xX, Y1, Y2, Y3, Y4


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

ensemble_predictor = Model([t_input, x_input], [y1, y2, y3, y4])
ensemble_predictor.summary()
ensemble_predictor.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(lr=1e-3))
init_weights = ensemble_predictor.get_weights()

for d in xrange(1, 32):
    callbacks = [
<<<<<<< HEAD
        CSVLogger('../results/sadHybridHumanPredictor/ensemble_predictor_2011_jan_v2/log_d{}.csv'.format(d), separator=',', append=False),
        ModelCheckpoint(filepath='../results/sadHybridHumanPredictor/ensemble_predictor_2011_jan_v2/ensemble_predictor_{}.hdf5'.format(d), verbose=1, save_best_only=True, monitor='loss'),
        # EarlyStopping(monitor='val_loss', patience=0, verbose=1, mode='auto')
        EarlyStopping(monitor='loss', patience=0, verbose=1, mode='auto')
=======
        CSVLogger('../results/sadHybridHumanPredictor/ensemble_predictor_2010_aug/log_d{}.csv'.format(d), separator=',', append=False),
        ModelCheckpoint(filepath='../results/sadHybridHumanPredictor/ensemble_predictor_2010_aug/ensemble_predictor_{}.hdf5'.format(d), verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=0, verbose=1, mode='auto')
>>>>>>> 010e667e618f2a9f320422a4045e99a8167f29ae
    ]
    tX, xX, Y1, Y2, Y3, Y4 = read_trainingset('/home/hpc/work/data/dis_forensemble_2010/', d)
    ensemble_predictor.set_weights(init_weights)
    # ensemble_predictor.fit([tX, xX], [Y1, Y2, Y3, Y4], batch_size=batch_size, epochs=20, shuffle=True,\
                            # validation_split=0.2, verbose=1, callbacks=callbacks)
    ensemble_predictor.fit([tX, xX], [Y1, Y2, Y3, Y4], batch_size=batch_size, epochs=20, shuffle=True,\
<<<<<<< HEAD
                            verbose=1, callbacks=callbacks)
=======
                            validation_split=0.2, verbose=1, callbacks=callbacks)
>>>>>>> 010e667e618f2a9f320422a4045e99a8167f29ae
