# -*- coding: utf-8 -*-
# @Time    : 17-8-13 上午10:13
# @Author  : FeiXue
# @Email   : feixue@pku.edu.cn
# @File    : plot_trajectory.py

import matplotlib.pyplot as plt
import math
import numpy as np
import sys


def load_data_quaternion(data, x, y, z):
    t = np.zeros(7)

    for line in data.readlines():
        line.strip()
        cols = line.split()
        for i in range(7):
            t[i] = float(cols[i])

        x.append(t[0])
        y.append(t[1])
        z.append(t[2])


def load_evaluation_kitti(data, x, y, z):
    t = np.zeros(12)

    for line in data.readlines():
        line.strip()
        cols = line.split()

        for i in range(12):
            t[i] = float(cols[i])

        x.append(t[3])
        y.append(t[7])
        z.append(t[11])


def load_data_gps(data, x, y):
    t = np.zeros(2)

    for line in data.readlines():
        line.strip()
        cols = line.split()
        for i in range(2):
            t[i] = float(cols[i])

        x.append(t[0])
        y.append(t[1])



x_e = []
y_e = []
z_e = []

x_g = []
y_g = []
z_g = []

data_gt = open(sys.argv[1], 'r')
load_evaluation_kitti(data_gt, x_g, y_g, z_g)
# load_data_gps(data_gt, x, y)

data_eva = open(sys.argv[2], 'r')
load_data_quaternion(data_eva, x_e, y_e, z_e)

# data_eva = open(sys.argv[1], 'r')
# load_data(data_eva, xp, yp, zp)

plt.plot(x_g, z_g, 'r.', label="groundtruth")
plt.plot(x_e, z_e, 'b.', label="evaluation")
plt.plot()
plt.legend()

plt.savefig('pku-kitti-00.png')
plt.show()