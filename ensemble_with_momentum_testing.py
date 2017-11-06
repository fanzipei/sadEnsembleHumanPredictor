#!/usr/bin/env python
# encoding: utf-8


from keras.models import Model, load_model
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.layers import GRU, Embedding, Input, RepeatVector, Dense, Flatten
from keras.layers.merge import concatenate
from keras.optimizers import RMSprop

import numpy as np

embedding_dim_time = 8
embedding_dim_loc = 64
hidden_dim = 128
# tokyo
# num_locs = 1441
# osaka
num_locs = 1537
batch_size = 256
T = 4


def read_trainingset(folderpath, d):
    xX = np.zeros([0, T], dtype=np.int32)
    Y1 = np.zeros([0], dtype=np.int32)
    filepath = '{}day_{}.csv'.format(folderpath, d)
    data = np.genfromtxt(filepath, dtype=np.int32, delimiter=',')

    X = []
    for t in xrange(1, 96 - T):
        xX = data[:, t + 1:t + 1 + T]
        Y1 = data[:, t + 1 + T]
        tX = np.array([t] * data.shape[0])
        X.append((xX, tX, Y1))

    return X


ensemble_predictor = load_model('../results/sadHybridHumanPredictor/online_predictor_with_momentum_2012_may_osaka/online_predictor_d{}t{}.hdf5'.format(1, 0))

open('../results/sadHybridHumanPredictor/ensemble_predictor_2012_may_osaka.csv', 'w').close()

for d in xrange(1, 32):
    X = read_trainingset('/home/fan/work/data/dis_forensemble_2012_may_osaka/', d)

    for t in xrange(96 - T - 1):
        ensemble_predictor.load_weights('../results/sadHybridHumanPredictor/online_predictor_with_momentum_2012_may_osaka/online_predictor_d{}t{}.hdf5'.format(d, t))

        xX, tX, Y1 = X[t]
        loss = ensemble_predictor.evaluate([tX, xX], Y1, batch_size=4096)
        print 'Day {}, t {}, loss {}'.format(d, t, loss)
        with open('../results/sadHybridHumanPredictor/ensemble_predictor_2012_may_osaka.csv', 'a') as f:
            f.write('{}\n'.format(loss))
