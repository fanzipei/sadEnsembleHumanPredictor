#!/usr/bin/env python
# encoding: utf-8

from keras.models import load_model

for i in xrange(1, 9):
    model = load_model('/home/fan/work/results/sadHybridHumanPredictor/ensemble_predictor_v1/ensemble_predictor_{}.hdf5'.format(i))
    model.save_weights('/home/fan/work/results/sadHybridHumanPredictor/ensemble_predictor_v1/ensemble_predictor_weights_{}.hdf5'.format(i))
