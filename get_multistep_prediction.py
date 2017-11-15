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
num_locs = 1441
# osaka
# num_locs = 1537
batch_size = 256
T = 4
MAX_STEP = 4
MCMC_SIZE = 16


def read_trainingset(folderpath, d):
    xX = np.zeros([0, T], dtype=np.int32)
    Y1 = np.zeros([0], dtype=np.int32)
    filepath = '{}day_{}.csv'.format(folderpath, d)
    data = np.genfromtxt(filepath, dtype=np.int32, delimiter=',')
    data = data[np.random.choice(data.shape[0], 50000, replace=False)]

    X = []
    for t in xrange(1, 96 - T - MAX_STEP + 1):
        xX = data[:, t + 1:t + 1 + T]
        Y1 = data[:, t + 1 + T]
        Y2 = data[:, t + 2 + T]
        Y3 = data[:, t + 3 + T]
        Y4 = data[:, t + 4 + T]
        tX = np.array([t] * data.shape[0])
        X.append((xX, tX, Y1, Y2, Y3, Y4))

    return X


ensemble_predictor = load_model('../results/sadHybridHumanPredictor/online_predictor_with_momentum_2012_may_tokyo/online_predictor_d{}t{}.hdf5'.format(1, 0))

open('../results/sadHybridHumanPredictor/multistep_2012_may_tokyo.csv', 'w').close()

for d in xrange(1, 32):
    X = read_trainingset('/home/fan/work/data/dis_forensemble_2012_may_tokyo/', d)

    for t in xrange(96 - T - MAX_STEP):
        ensemble_predictor.load_weights('../results/sadHybridHumanPredictor/online_predictor_with_momentum_2012_may_tokyo/online_predictor_d{}t{}.hdf5'.format(d, t))

        xX, tX, Y1, Y2, Y3, Y4 = X[t]
        loss1 = ensemble_predictor.evaluate([tX, xX], Y1, batch_size=8192)
        next_step_1 = ensemble_predictor.predict([tX, xX], batch_size=8192)
        loss2_sum = 0
        loss3_sum = 0
        loss4_sum = 0
        for i in xrange(MCMC_SIZE):
            print 'MCMC iteration {}'.format(i)
            next_step_samples = np.expand_dims(np.array(map(lambda p: np.random.choice(range(num_locs), p=p), next_step_1)), -1)
            print next_step_samples.shape
            xX_step1 = np.concatenate([xX[:, 1:], next_step_samples], axis=1)
            loss2_sum += ensemble_predictor.evaluate([tX + 1, xX_step1], Y2, batch_size=8192)

            next_step_2 = ensemble_predictor.predict([tX + 1, xX_step1], batch_size=8192)
            next_step_samples = np.expand_dims(np.array(map(lambda p: np.random.choice(range(num_locs), p=p), next_step_2)), -1)
            xX_step2 = np.concatenate([xX_step1[:, 1:], next_step_samples], axis=1)
            loss3_sum += ensemble_predictor.evaluate([tX + 2, xX_step2], Y3, batch_size=8192)

            next_step_3 = ensemble_predictor.predict([tX + 2, xX_step2], batch_size=8192)
            next_step_samples = np.expand_dims(np.array(map(lambda p: np.random.choice(range(num_locs), p=p), next_step_3)), -1)
            xX_step3 = np.concatenate([xX_step2[:, 1:], next_step_samples], axis=1)
            loss4_sum += ensemble_predictor.evaluate([tX + 3, xX_step3], Y4, batch_size=8192)

        loss2 = loss2_sum / MCMC_SIZE
        loss3 = loss3_sum / MCMC_SIZE
        loss4 = loss4_sum / MCMC_SIZE
        print loss1
        print loss2
        print loss3
        print loss4
        open('../results/sadHybridHumanPredictor/multistep_2012_may_tokyo.csv', 'a').write('{},{},{},{}\n'.format(loss1, loss2, loss3, loss4))
