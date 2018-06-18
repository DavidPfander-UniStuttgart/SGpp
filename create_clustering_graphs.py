#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt


import csv
f = open('toy_density_eval.csv', 'r')
reader = csv.reader(f)
content = [row for row in reader]
# for row in reader:
#     print row

X = [row[0] for row in content]
Y = [row[1] for row in content]
DensityValues = [row[2] for row in content]
print("X:")
print(X)
print("Y:")
print(Y)
print("Values:")
print(DensityValues)

colors = ["r", "r", "r"]
markers = ["o", "o", "o"]
c=colors[0]
m=markers[0]

##################################################
# Estimate density
##################################################

fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
# ax.set_zlim(0, 1)
# ax.set_zticks([])

# plt.imshow(sum_of_results, extent=(0.0, 1.0, 0.0, 1.0), origin='lower', interpolation='gaussian', alpha=1.0, cmap='viridis', aspect='auto') # cmap='hot', , interpolation='nearest'

# # Plot a basic wireframe
# ax.plot_wireframe(*overall_distribution_grid, alpha=0.3)

# if add_noise:
#     ax.scatter(noise_dataset[0], noise_dataset[1], c='b')

# for i in range(2):
#     c=colors[i]
#     m=markers[i]
# ax.scatter(X, Y, c=c, marker=m, zorder=2)

# ax.scatter(X, Y, c=DensityValues, zorder=2)

ax.contourf(X, Y, c=DensityValues) #, zorder=2

# # ax.view_init(elev=90, azim=5)
# plt.show()
fig.savefig("clustering_2.png", dpi=300)

plt.close()
