#!/usr/bin/env python
# encoding: utf-8

import csv

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


def output_traj(in_filename, out_filename, user_set):
    with open(in_filename, 'r') as fin:
        with open(out_filename, 'w') as fout:
            for uid_str, time_str, lat_str, lon_str, _, _, in csv.reader(fin):
                uid = int(uid_str)
                if uid in user_set:
                    if user_set[uid] > 5:
                        fout.write('{},{},{},{}\n'.format(uid_str, time_str, lat_str, lon_str))
                    else:
                        del user_set[uid]


def main():
    lat_min = 35.5
    lat_max = 35.8
    lon_min = 139.4
    lon_max = 139.9
    in_filename = '/media/fan/HDPC-UT/ZDC/2012/FeaturePhone/20120402.csv'
    user_set = filter_users_in_region(in_filename, lat_min, lat_max, lon_min, lon_max)
    out_filename = 'usersintokyo_20120402.csv'
    output_traj(in_filename, out_filename, user_set)


if __name__ == '__main__':
    main()
