#!/usr/bin/env python
# encoding: utf-8

from keras import backend as K
from keras.layers.merge import dot, concatenate
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.layers import GRU, Embedding, Input, RepeatVector, Dense, Flatten, Reshape
from keras.models import Model, load_model

import numpy as np

embedding_dim_time = 8
embedding_dim_loc = 64
num_models = 3
hidden_dim = 128
num_locs = 1441
batch_size = 256
T = 4

# x1 = Input((5, 1))
# x2 = Input((5, 1))
# x = concatenate([x1, x2], axis=-1)
# w = Input((2, 1))
# y = dot([x, w], [2, 1])
# Model([x1, x2, w], y).summary()

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


def build_and_load_model(model_path):
    print 'Build model {}'.format(model_path)
    t_input = Input(shape=(1,))
    x_input = Input(shape=(T,))
    temb = Flatten()(Embedding(96, embedding_dim_time)(t_input))
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
    predictor.trainable = False
    predictor.load_weights(model_path)

    return predictor


models = [build_and_load_model('../results/sadHybridHumanPredictor/ensemble_predictor_v1/ensemble_predictor_weights_{}.hdf5'.format(i)) for i in xrange(1, num_models + 1)]
print 'Load Models Finished'

t_input = Input(shape=(1,))
x_input = Input(shape=(T,))
temb = Flatten()(Embedding(96, embedding_dim_time)(t_input))
xemb = Embedding(num_locs, embedding_dim_loc, input_length=T)(x_input)
rep_time = RepeatVector(T)(temb)
merge_input = concatenate([rep_time, xemb], axis=-1)
gru1 = GRU(hidden_dim, return_sequences=True, unroll=True, activation='softsign')(merge_input)
models_out = [[Reshape((num_locs, 1))(model([t_input, x_input])[i]) for model in models] for i in xrange(4)]
preds = [concatenate(models_out[i], axis=-1) for i in xrange(4)]
weights = Dense(num_models, activation='softmax')(gru1)
y1 = dot([preds[0], weights], [2, 2])
y2 = dot([preds[1], weights], [2, 2])
y3 = dot([preds[2], weights], [2, 2])
y4 = dot([preds[3], weights], [2, 2])
online_predictor = Model([t_input, x_input], [y1, y2, y3, y4])
online_predictor.summary()


# for d in xrange(32, 62):
    # callbacks = [
        # CSVLogger('../results/sadHybridHumanPredictor/online_predictor/log_d{}.csv'.format(d), separator=',', append=False),
        # ModelCheckpoint(filepath='../results/sadHybridHumanPredictor/online_predictor/ensemble_predictor_{}.hdf5'.format(d), verbose=1, save_best_only=True),
        # EarlyStopping(monitor='val_loss', patience=0, verbose=1, mode='auto')
    # ]
    # tX, xX, Y1, Y2, Y3, Y4 = read_trainingset('/home/fan/work/data/dis_forensemble/', d)
    # online_predictor.fit([tX, xX], [Y1, Y2, Y3, Y4], batch_size=batch_size, epochs=20, shuffle=True,\
                            # validation_split=0.2, verbose=1, callbacks=callbacks)
