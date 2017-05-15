#!/usr/bin/env python
# encoding: utf-8

import os
import csv
import time
import meshlonlat
import numpy as np


start_time = time.mktime(time.strptime('2012-07-31 23:50:00', '%Y-%m-%d %H:%M:%S'))
loc_dict = dict({})
lat_min = 35.5
lat_max = 35.8
lon_min = 139.4
lon_max = 139.9


def filename_generator(folder_path):
    filename = '2012{:02d}{:02d}.csv'
    day_idx = 1
    for m in xrange(8, 9):
        for d in xrange(1, 32):
            full_path = os.path.join(folder_path, filename.format(m, d))
            if os.path.isfile(full_path):
                yield day_idx, full_path
                day_idx += 1


def read_traj(filename, start_time):
    user_traj = dict({})
    with open(filename, 'r') as f:
        for uid_str, time_str, lat_str, lon_str in csv.reader(f):
            tstamp = time.mktime(time.strptime(time_str, '%Y-%m-%d %H:%M:%S')) - start_time
            if tstamp < - 600 or tstamp >= 24 * 3600:
                continue
            lat = float(lat_str)
            lon = float(lon_str)
            # uid = int(uid_str[3:])
            uid = int(uid_str)
            if uid not in user_traj:
                user_traj[uid] = []
            user_traj[uid].append((tstamp, lat, lon))

    for uid in user_traj:
        user_traj[uid] = sorted(user_traj[uid], key=lambda x:x[0])

    return user_traj


def get_current_latlon(traj_raw, t):
    if t < traj_raw[0][0]:
        return traj_raw[0][1], traj_raw[0][2]

    for i in xrange(len(traj_raw) - 1):
        pre_t = traj_raw[i][0]
        pre_lat = traj_raw[i][1]
        pre_lon = traj_raw[i][2]
        pro_t = traj_raw[i + 1][0]
        pro_lat = traj_raw[i + 1][1]
        pro_lon = traj_raw[i + 1][2]
        if pre_t < t and pro_t >= t:
            d_lat = pro_lat - pre_lat
            d_lon = pro_lon - pre_lon
            d_t = pro_t - pre_t
            if d_t < 1e-6:
                return 0.5 * (pre_lat + pro_lat), 0.5 * (pre_lon + pro_lon)
            else:
                return pre_lat + (t - pre_t) / d_t * d_lat, pre_lon + (t - pre_t) / d_t * d_lon

    return traj_raw[-1][1], traj_raw[-1][2]


def get_training_set(user_traj):
    user_traj_matrix = np.zeros([len(user_traj), 97], dtype=np.int32)
    uidx = 0
    for uid in user_traj:
        user_traj_matrix[uidx, 0] = uid
        for t in xrange(96):
            cur_lat, cur_lon = get_current_latlon(user_traj[uid], t * 900)
            meshcode = meshlonlat.lonlat2mesh(cur_lon, cur_lat, 1000)
            lidx = 0
            if meshcode in loc_dict:
                lidx = loc_dict[meshcode]
            user_traj_matrix[uidx, t + 1] = lidx
        uidx += 1

    return user_traj_matrix


# dlat = (lat_max - lat_min) * 0.01
# dlon = (lon_max - lon_min) * 0.01
# for i in xrange(100):
    # for j in xrange(100):
        # lat = lat_min + dlat * (i + 0.5)
        # lon = lon_min + dlon * (j + 0.5)
        # meshcode = meshlonlat.lonlat2mesh(lon, lat, 1000)
        # if meshcode not in loc_dict:
            # loc_dict[meshcode] = len(loc_dict)

# print 'Initialize Loc dictionary'
# with open('/home/fan/work/data/dis_forensemble/loc_dict.csv', 'w') as f:
    # for meshcode in loc_dict:
        # f.write('{},{}\n'.format(meshcode, loc_dict[meshcode] + 1))

print 'Initialize loc dictionary from file'
with open('./loc_dict.csv', 'r') as f:
    for meshcode, lidx in csv.reader(f):
        loc_dict[meshcode] = int(lidx)

for day_idx, filename in filename_generator('/home/fan/work/data/UsersInTokyo/'):
    # if day_idx < 19:
        # continue
    print 'Read {}'.format(filename)
    user_traj = read_traj(filename, start_time + (day_idx - 1) * 3600 * 24)
    print 'Number of users: {}'.format(len(user_traj))
    training_set = get_training_set(user_traj)
    np.savetxt('/home/fan/work/data/dis_forensemble_2012_aug/day_{}.csv'.format(day_idx), training_set, delimiter=',', fmt='%i')
