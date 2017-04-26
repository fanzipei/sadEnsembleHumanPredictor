#!/usr/bin/env python
# encoding: utf-8

import csv
import os

def filter_users_in_region(filename, lat_min, lat_max, lon_min, lon_max):
    user_set = dict({})
    with open(filename, 'r') as f:
        for uid_str, time_str, lat_str, lon_str, _, _, in csv.reader(f):
            uid = int(uid_str)
            lat = float(lat_str)
            lon = float(lon_str)
            if lat > lat_min and lat < lat_max and lon > lon_min and lon < lon_max:
                if uid not in user_set:
                    user_set[uid] = 1
                else:
                    user_set[uid] += 1

    return user_set


def output_traj(full_path, out_folder, out_filename, user_set):
    with open(full_path, 'r') as fin:
        with open(os.path.join(out_folder, out_filename), 'w') as fout:
            for uid_str, time_str, lat_str, lon_str, _, _, in csv.reader(fin):
                uid = int(uid_str)
                if uid in user_set:
                    if user_set[uid] > 5:
                        fout.write('{},{},{},{}\n'.format(uid_str, time_str, lat_str, lon_str))
                    else:
                        del user_set[uid]


def filename_generator(folder_path):
    filename_fmt = '2012{:02d}{:02d}.csv'
    for m in xrange(5, 6):
        for d in xrange(1, 32):
            filename = filename_fmt.format(m, d)
            full_path = os.path.join(folder_path, filename)
            print 'Reading {}'.format(full_path)
            if os.path.isfile(full_path):
                yield filename, full_path


def main():
    lat_min = 35.5
    lat_max = 35.8
    lon_min = 139.4
    lon_max = 139.9
    folder_path = '/media/fan/HDPC-UT/ZDC/2012/FeaturePhone/'
    for filename, full_path in filename_generator(folder_path):
        user_set = filter_users_in_region(full_path, lat_min, lat_max, lon_min, lon_max)
        out_folder = '/media/fan/HDPC-UT/ZDC/TrainingForMapping/usersintokyo/'
        output_traj(full_path, out_folder, filename, user_set)


if __name__ == '__main__':
    main()
