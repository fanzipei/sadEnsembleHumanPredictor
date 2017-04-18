#!/usr/bin/env python
# encoding: utf-8


def lonlat2mesh(x, y, meshsize = 1000):
    y = float(y) * 1.5
    x = float(x) - 100

    mesh1_y = int(y)
    mesh1_x = int(x)

    tmp_y = (y - mesh1_y) / (10.0 / 8.0)
    tmp_x = (x - mesh1_x) / (10.0 / 8.0)

    mesh2_y = int(tmp_y * 10)
    mesh2_x = int(tmp_x * 10)

    tmp_y = (tmp_y * 10 - mesh2_y)
    tmp_x = (tmp_x * 10 - mesh2_x)

    mesh3_y = int(tmp_y * 10)
    mesh3_x = int(tmp_x * 10)

    if meshsize == 1000:
        return '%02d%02d%d%d%d%d' % (mesh1_y, mesh1_x, mesh2_y, mesh2_x, mesh3_y, mesh3_x)

    tmp_y = (tmp_y * 10 - mesh3_y)
    tmp_x = (tmp_x * 10 - mesh3_x)

    if tmp_y >= 0.5:
        if tmp_x >= 0.5:
            mesh4 = 4
            tmp_x -= 0.5
        else:
            mesh4 = 3
        tmp_y -= 0.5
    else:
        if tmp_x >= 0.5:
            mesh4 = 2
            tmp_x -= 0.5
        else:
            mesh4 = 1

    if meshsize == 250 or meshsize == 50 or meshsize == 25:
        div_y = tmp_y // 0.25
        div_x = tmp_x // 0.25

        mesh5 = int(div_x + div_y * 2 + 1)
        if tmp_y >= 0.25:
            tmp_y -= 0.25
        if tmp_x >= 0.25:
            tmp_x -= 0.25

        if meshsize == 50 or meshsize == 25:
            div_y = tmp_y // 0.05
            div_x = tmp_x // 0.05

            mesh50 = int(div_x + div_y * 5 + 1)
            if tmp_x >= 0.05 * div_x:
                tmp_x -= 0.05 * div_x
            if tmp_y >= 0.05 * div_y:
                tmp_y -= 0.05 * div_y

            if meshsize == 25:
                div_y = tmp_y // 0.025
                div_x = tmp_x // 0.025
                mesh25 = int(div_x + div_y * 2 + 1)

    elif meshsize == 100:
        div_y = tmp_y // 0.1
        div_x = tmp_x // 0.1

        mesh100 = int(div_x + div_y * 5 + 1)

    if meshsize == 250:
        result = '%02d%02d%d%d%d%d%d%d' % (mesh1_y, mesh1_x, mesh2_y, mesh2_x, mesh3_y, mesh3_x, mesh4, mesh5)
    elif meshsize == 100:
        result = '%02d%02d%d%d%d%d%d%02d' % (mesh1_y, mesh1_x, mesh2_y, mesh2_x, mesh3_y, mesh3_x, mesh4, mesh100)
    elif meshsize == 50:
        result = '%02d%02d%d%d%d%d%d%d%02d' % (mesh1_y, mesh1_x, mesh2_y, mesh2_x, mesh3_y, mesh3_x, mesh4, mesh5, mesh50)
    elif meshsize == 25:
        result = '%02d%02d%d%d%d%d%d%d%02d%d' % (mesh1_y, mesh1_x, mesh2_y, mesh2_x, mesh3_y, mesh3_x, mesh4, mesh5, mesh50, mesh25)
    else:
        result = '%02d%02d%d%d%d%d%d' % (mesh1_y, mesh1_x, mesh2_y, mesh2_x, mesh3_y, mesh3_x, mesh4)

    return result


def mesh2lonlat(mesh, meshsize = 1000, is_center = False):
    mesh1_y = float(mesh[:2]) / 1.5
    mesh1_x = float(mesh[2:4]) + 100
    mesh2_y = float(mesh[4:5]) * 5
    mesh2_x = float(mesh[5:6]) * 7.5
    mesh3_y = float(mesh[6:7]) * 30
    mesh3_x = float(mesh[7:8]) * 45

    if meshsize < 1000:
        mesh4 = int(mesh[8:9])

        if mesh4 == 1:
            mesh4_y = 30.0 * 0
            mesh4_x = 45.0 * 0
        elif mesh4 == 2:
            mesh4_y = 30.0 * 0
            mesh4_x = 45.0 * 0.5
        elif mesh4 == 3:
            mesh4_y = 30.0 * 0.5
            mesh4_x = 45.0 * 0
        else:
            mesh4_y = 30.0 * 0.5
            mesh4_x = 45.0 * 0.5

    center_y = 7.5
    center_x = 11.25

    if meshsize == 250 or meshsize == 50 or meshsize == 25:
        mesh5 = int(mesh[9:10]) - 1
        div_y = mesh5 / 2
        div_x = mesh5 % 2
        mesh5_y = 7.5 * float(div_y)
        mesh5_x = 11.25 * float(div_x)
        center_y = 3.75
        center_x = 5.625

        if meshsize == 50 or meshsize == 25:
            mesh50 = int(mesh[10:12]) - 1
            div_y = mesh50 / 5
            div_x = mesh50 % 5
            mesh50_y = 1.5 * float(div_y)
            mesh50_x = 2.25 * float(div_x)
            center_y = 0.75
            center_x = 1.125

            if meshsize == 25:
                mesh25 = int(mesh[12:13]) - 1
                div_y = mesh25 / 2
                div_x = mesh25 % 2
                mesh25_y = 0.75 * float(div_y)
                mesh25_x = 1.125 * float(div_x)
                center_y = 0.375
                center_x = 0.5625

    if meshsize == 100:
        mesh100 = int(mesh[9:11]) - 1
        div_y = mesh100 / 5
        div_x = mesh100 % 5
        mesh100_y = 3.0 * float(div_y)
        mesh100_x = 4.5 * float(div_x)
        center_y = 1.5
        center_x = 2.25

    if not is_center:
        center_y = 0.0
        center_x = 0.0

    if meshsize == 500:
        lat = float(mesh1_y * 3600 + mesh2_y * 60 + mesh3_y + mesh4_y + center_y) / 3600
        lon = float(mesh1_x * 3600 + mesh2_x * 60 + mesh3_x + mesh4_x + center_x) / 3600
    elif meshsize == 250:
        lat = float(mesh1_y * 3600 + mesh2_y * 60 + mesh3_y + mesh4_y + mesh5_y + center_y) / 3600
        lon = float(mesh1_x * 3600 + mesh2_x * 60 + mesh3_x + mesh4_x + mesh5_x + center_x) / 3600
    elif meshsize == 100:
        lat = float(mesh1_y * 3600 + mesh2_y * 60 + mesh3_y + mesh4_y + mesh100_y + center_y) / 3600
        lon = float(mesh1_x * 3600 + mesh2_x * 60 + mesh3_x + mesh4_x + mesh100_x + center_x) / 3600
    elif meshsize == 50:
        lat = float(mesh1_y * 3600 + mesh2_y * 60 + mesh3_y + mesh4_y + mesh5_y + mesh50_y + center_y) / 3600
        lon = float(mesh1_x * 3600 + mesh2_x * 60 + mesh3_x + mesh4_x + mesh5_x + mesh50_x + center_x) / 3600
    elif meshsize == 25:
        lat = float(mesh1_y * 3600 + mesh2_y * 60 + mesh3_y + mesh4_y + mesh5_y + mesh50_y + mesh25_y + center_y) / 3600
        lon = float(mesh1_x * 3600 + mesh2_x * 60 + mesh3_x + mesh4_x + mesh5_x + mesh50_x + mesh25_x + center_x) / 3600
    else:
        lat = float(mesh1_y * 3600 + mesh2_y * 60 + mesh3_y + center_y) / 3600
        lon = float(mesh1_x * 3600 + mesh2_x * 60 + mesh3_x + center_x) / 3600

    return lat, lon
