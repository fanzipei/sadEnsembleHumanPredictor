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
weights_predictor = Model([t_input, x_input], weights)

online_predictor.load_weights('/home/hpc/work/results/sadHybridHumanPredictor/online_predictor_2012_aug/online_predictor_d{}t{}.hdf5'.format(4, 22))
tX = np.zeros([10, 1], dtype=np.int) + 26
xX = np.array([[1360, 1193, 1030, 789]], dtype=np.int).repeat(10, axis=0)
# pred = online_predictor.predict([tX, xX])

# output_matrix = np.zeros([10000, 97], dtype=np.int)
# output_matrix[:, 22:26] = xX

# for i in xrange(4):
    # for j in xrange(10000):
        # p = pred[i][j, :]
        # output_matrix[j, 26+i] = np.random.choice(num_locs, 1, p=p)[0]

# np.savetxt('noncomiket_pred.csv', output_matrix, delimiter=',', fmt='%i')

weights = weights_predictor.predict([tX, xX])[0]
np.savetxt('weights_noncomiket.csv', weights, delimiter=',')

online_predictor.load_weights('/home/hpc/work/results/sadHybridHumanPredictor/online_predictor_2012_aug/online_predictor_d{}t{}.hdf5'.format(11, 22))
weights = weights_predictor.predict([tX, xX])[0]
np.savetxt('weights_comiket.csv', weights, delimiter=',')
