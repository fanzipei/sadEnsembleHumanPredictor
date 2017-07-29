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


# online_predictor = load_model('/home/fan/work/results/sadHybridHumanPredictor/online_predictor/online_predictor_d32t0.hdf5')
# online_predictor.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(lr=1e-3))
# print 'Load ensemble predictor finished'
models = [build_and_load_model('../results/sadHybridHumanPredictor/ensemble_predictor_2010_aug/ensemble_predictor_{}.hdf5'.format(i)) for i in xrange(1, num_models + 1)]
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

tX = None
xX = None
Y1 = None
Y2 = None
Y3 = None
Y4 = None
data = np.genfromtxt('', dtype=np.int32, delimiter=',')
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

pd1 = np.zeros(89)
pd2 = np.zeros(89)
pd3 = np.zeros(89)
pd4 = np.zeros(89)
for t in xrange(89):
    online_predictor.load_weights('/home/hpc/work/results/sadHybridHumanPredictor/online_predictor_2012_aug/online_predictor_d{}t{}.hdf5'.format(11, t))
    Y = online_predictor.predict([tX, xX])
    pd1[t] = np.sum(Y[0][:, 632])
    pd2[t] = np.sum(Y[1][:, 632])
    pd3[t] = np.sum(Y[2][:, 632])
    pd4[t] = np.sum(Y[3][:, 632])

gtpd1 = Y1.shape[0] - np.count_nonzero(Y1 - 632, axis=0)
gtpd2 = Y2.shape[0] - np.count_nonzero(Y2 - 632, axis=0)
gtpd3 = Y3.shape[0] - np.count_nonzero(Y3 - 632, axis=0)
gtpd4 = Y4.shape[0] - np.count_nonzero(Y4 - 632, axis=0)

out = np.array([pd1, pd2, pd3, pd4, gtpd1, gtpd2, gtpd3, gtpd4])
np.savetxt('population_density.csv', out, delimiter=',')
