#!/usr/bin/env python
# encoding: utf-8

from keras import backend as K
from keras.layers.merge import dot, concatenate
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.layers import GRU, Embedding, Input, RepeatVector, Dense, Flatten, Reshape
from keras.models import Model, load_model
from keras.optimizers import RMSprop
import numpy as np
import meshlonlat
import csv


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

    return predictor


models = [build_and_load_model('../results/sadHybridHumanPredictor/ensemble_predictor/ensemble_predictor_{}.hdf5'.format(i)) for i in xrange(1, num_models + 1)]
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
online_predictor.load_weights('/home/fan/work/results/sadHybridHumanPredictor/online_predictor_2012_jan/online_predictor_d1t0.hdf5')
online_predictor_weights_output = Model(x_input, weights)
online_predictor_weights_output.summary()
online_predictor_weights_output.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(lr=1e-3))

X = read_trainingset('/home/fan/work/data/dis_forensemble_2012_jan/', 1)
pred_weights = online_predictor_weights_output.predict(X[0][1])
print pred_weights
print pred_weights.shape
pred_weights = pred_weights.squeeze()
print pred_weights.shape
# np.savetxt('weights.csv', pred_weights, delimiter=',')

idx2mesh = dict({})
with open('/home/fan/work/data/dis_forensemble/loc_dict.csv', 'r') as f:
    for meshcode, lidx_str in csv.reader(f):
        idx2mesh[int(lidx_str)] = meshcode

timestr = ['2012-01-01 00:00:00', '2012-01-01 00:15:00', '2012-01-01 00:30:00', '2012-01-01 00:45:00']
cluster = np.argmax(pred_weights, axis=1)
data = X[0][1]
with open('test.csv', 'w') as f:
    for i in xrange(data.shape[0]):
        for t in xrange(T):
            if data[i, t] == 0:
                continue
            else:
                lon, lat = meshlonlat.mesh2lonlat(idx2mesh[data[i, t]], 1000, is_center=True)
                lon += np.random.ranf() * 0.008 - 0.004
                lat += np.random.ranf() * 0.010 - 0.005
                f.write('{},{},{},{},{}\n'.format(i, timestr[t], lon, lat, cluster[i]))

for d in xrange(num_models):
    with open('../results/sadHybridHumanPredictor/model_selection/model_selection_{}.csv'.format(d), 'w') as f:
        for i in xrange(data.shape[0]):
            for t in xrange(T):
                if data[i, t] == 0:
                    continue
                else:
                    lon, lat = meshlonlat.mesh2lonlat(idx2mesh[data[i, t]], 1000, is_center=True)
                    lon += np.random.ranf() * 0.008 - 0.004
                    lat += np.random.ranf() * 0.010 - 0.005
                    f.write('{},{},{},{},{}\n'.format(i, timestr[t], lon, lat, int(pred_weights[i][d] * 30)))
