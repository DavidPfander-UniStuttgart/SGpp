#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

l = 1
i = 1

def hat1d(x):
    global l
    global i
    # return max(0, 1 - abs(2**l * x - i))
    temp = 1 - np.abs(2 * x - 1)
    # print(temp)
    # return np.max(0, temp)
    return temp

def hat2d(x0, x1):
    return hat1d(x0) * hat1d(x1)

x0 = np.linspace(0.0, 1.0, 20 + 1)
x1 = np.linspace(0.0, 1.0, 20 + 1)
gridX, gridY = np.meshgrid(x0, x1)
print(gridX)
print(gridY)
evals = hat2d(gridX, gridY)
print(evals)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(gridX, gridY, evals)
# plt.show()
plt.savefig("graphs/hat_2d.eps")
# y = [[hat2d(x0[i], x1[j]) for j in range(len(x1)] for i in range(len(x0))]
# print(y)
# print(mesh)
