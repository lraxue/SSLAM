# -*- coding: utf-8 -*-
# @Time    : 17-8-13 上午10:15
# @Author  : FeiXue
# @Email   : feixue@pku.edu.cn
# @File    : plot_reprojection_error.py

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import sys
import math

def load_error(data):
    W = 1241
    H = 376
    t = np.zeros(19)   # load with, height, error, x, y
    x = []
    y = []
    z = []
    d = []
    xz = np.zeros(W)
    yz = np.zeros(H)
    n_xz = np.zeros(W)
    n_yz = np.zeros(H)


    for line in data.readlines():
        line.strip()
        values = line.split()

        if len(values) < 19:
            continue

        for i in range (19):
            t[i] = float(values[i])

        if t[2] > 20:
            continue

        z.append(t[2])
        x.append(t[3])
        y.append(t[4])

        xz[int(t[3])] += t[2]
        yz[int(t[4])] += t[2]
        n_xz[int(t[3])] += 1
        n_yz[int(t[4])] += 1

        dist = math.sqrt((t[3] - 607) * (t[3] - 607) + (t[4] - 185) * (t[4] - 185))
        d.append(dist)

    for i in range (W):
        if n_xz[i] > 0:
            xz[i] /= n_xz[i]

    for i in range(H):
        if n_yz[i] > 0:
            yz[i] /= n_yz[i]

    return x, y, z, d, xz, yz


data = open('error.txt', 'r')
x, y, z, d, xz, yz = load_error(data)
# print(x, y, z)

# ax = plt.figure().add_subplot(111, projection='3d')
# ax.scatter(x, y, z, c='r', marker='^')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z_Label')
# plt.show()

plt.figure()
plt.plot(x, z, 'o', color='m')
plt.title("x_z")
plt.show()

plt.figure()
plt.plot(y, z, 'o', color='r')
plt.title("y_z")
plt.show()

plt.figure()
plt.plot(d, z, 'o', color='r')
plt.title("d_z")
plt.show()

plt.figure()
plt.plot(xz, 'o', color='r')
plt.title("xz")
plt.show()

plt.figure()
plt.plot(yz, 'o', color='r')
plt.title("yz")
plt.show()

plt.figure()
plt.hist(z, color='r')
plt.title("hist_z")
plt.show()






