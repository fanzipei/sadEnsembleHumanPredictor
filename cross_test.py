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
num_locs = 1441
batch_size = 8192
T = 4

def read_trainingset(folderpath, d):
    tX = None
    xX = None
    Y1 = None
    Y2 = None
    Y3 = None
    Y4 = None
    filepath = '{}day_{}.csv'.format(folderpath, d)
    print 'Read {}'.format(filepath)
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

    choice = np.random.choice(tX.shape[0], 1000000, replace=False)

    return tX[choice], xX[choice], Y1[choice], Y2[choice], Y3[choice], Y4[choice]


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
    predictor.trainable = False
    predictor.load_weights(model_path)
    predictor.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(lr=1e-3))

    return predictor


models = [build_and_load_model('../results/sadHybridHumanPredictor/ensemble_predictor_2011_jan/ensemble_predictor_{}.hdf5'.format(i)) for i in xrange(1, num_models + 1)]
print 'Load Models Finished'

X = [(read_trainingset('../data/dis_forensemble_2011_jan/', d)) for d in xrange(1, num_models + 1)]
eval_matrix = np.zeros([num_models, num_models])
for i in xrange(num_models):
    for j in xrange(num_models):
        print 'Evaluate model {} on day {}'.format(i, j)
        eval_matrix[i, j] = models[i].evaluate([X[j][0], X[j][1]], [X[j][2], X[j][3], X[j][4], X[j][5]], batch_size=batch_size)[0]
        # print models[i].evaluate([X[j][0], X[j][1]], [X[j][2], X[j][3], X[j][4], X[j][5]], batch_size=batch_size)

np.savetxt('eval_matrix.csv', eval_matrix, delimiter=',')
