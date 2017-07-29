#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import meshlonlat
import csv
import time


data = np.genfromtxt('./noncomiket_pred.csv', delimiter=',', dtype=np.int32)

idx2mesh = dict({})
with open('./loc_dict.csv', 'r') as f:
    for meshcode, lidx_str in csv.reader(f):
        idx2mesh[int(lidx_str)] = meshcode

start_time = time.mktime(time.strptime('2012-05-01 23:50:00','%Y-%m-%d %H:%M:%S'))
with open('test_noncomiket_pred.csv', 'w') as f:
    f.write('id,time,lon,lat\n')
    for i in xrange(data.shape[0]):
        for j in xrange(1, data.shape[1]):
            if data[i, j] == 0:
                continue
            else:
                lon, lat = meshlonlat.mesh2lonlat(idx2mesh[data[i, j]], 1000, is_center=True)
                lon += np.random.ranf() * 0.008 - 0.004
                lat += np.random.ranf() * 0.010 - 0.005
                time_str = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(start_time + j * 900))
                f.write('{},{},{},{}\n'.format(i, time_str, lon, lat))
