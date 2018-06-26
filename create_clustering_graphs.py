#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import csv
import argparse

parser = argparse.ArgumentParser(description='Create graphs.')
# parser.add_argument('--eval_level', dest='eval_level', action='store_const')
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument('--eval_grid_level', type=int, required=True)
requiredNamed.add_argument('--scenario_name', required=True)
requiredNamed.add_argument('--dataset_name', required=True)
parser.add_argument('--use_unpruned_graph', default=False, action='store_true')
parser.add_argument('--dont_display_grid', default=False, action='store_true')
parser.add_argument('--dont_display_knn', default=False, action='store_true')
args = parser.parse_args()

# read-in dataset
f = open(args.dataset_name, 'r')
reader = csv.reader(f)
content = [] # [row for row in reader]
found_data_keyword = False
for c in reader:
    if not found_data_keyword:
        if len(c) > 0 and c[0] == "@DATA":
            found_data_keyword = True
        continue
    content += [c]
X_dataset = [float(row[0]) for row in content]
Y_dataset = [float(row[1]) for row in content]

# read-in grid points
f = open("results/" + args.scenario_name + "_grid.csv", 'r')
reader = csv.reader(f)
content = [] # [row for row in reader]
for c in reader:
    content += [c]
X_grid = [float(row[0]) for row in content]
Y_grid = [float(row[1]) for row in content]

# read-in full-grid-evaluated density function
f = open("results/" + args.scenario_name + '_density_eval.csv', 'r')
reader = csv.reader(f)
content = [row for row in reader]

print("eval_grid_level:", args.eval_grid_level)
eval_h = 1.0 / pow(2.0, args.eval_grid_level) # 2^-eval_grid_level
print("eval_h:", eval_h)
eval_dim_grid_points = (1 << args.eval_grid_level) + 1;
print("eval_dim_grid_points:", eval_dim_grid_points)

X_temp = [row[0] for row in content]
X_density_grid = []
for r in range(eval_dim_grid_points):
    row = []
    for c in range(eval_dim_grid_points):
        row += [float(X_temp[r * eval_dim_grid_points + c])]
    X_density_grid += [row]

Y_temp = [row[1] for row in content]
Y_density_grid = []
for r in range(eval_dim_grid_points):
    row = []
    for c in range(eval_dim_grid_points):
        row += [float(Y_temp[r * eval_dim_grid_points + c])]
    Y_density_grid += [row]

DensityValues_temp = [row[2] for row in content]
DensityValues = []
for r in range(eval_dim_grid_points):
    row = []
    for c in range(eval_dim_grid_points):
        row += [float(DensityValues_temp[r * eval_dim_grid_points + c])]
    DensityValues += [row]
# print("X_density_grid:")
# print(X_density_grid)
# print("Y_density_grid:")
# print(Y_density_grid)
# print("Values:")
# print(DensityValues)


# read-in dataset
f = open("results/" + args.scenario_name + "_graph.csv", 'r')
reader = csv.reader(f)
neighborhood_list = [[int(i) for i in row] for row in reader]
# print(neighborhood_list)
graph_edges = []
for l_index in range(len(neighborhood_list)):
    # print(len(neighborhood_list[l_index]))
    for n_index in neighborhood_list[l_index]:
        print("from: " + str(l_index) + " -> " + str(n_index))

        line = [[X_dataset[l_index], Y_dataset[l_index]], [X_dataset[n_index], Y_dataset[n_index]]]
        graph_edges += [line]
graph_edges_collection = LineCollection(graph_edges, linewidths=(1), colors = ['r'], linestyle='solid')

# read-in pruned dataset
f = open("results/" + args.scenario_name + "_graph_pruned.csv", 'r')
reader = csv.reader(f)
pruned_neighborhood_list = [[int(i) for i in row] for row in reader]
# print(neighborhood_list)
pruned_graph_edges = []
for l_index in range(len(pruned_neighborhood_list)):
    for n_index in pruned_neighborhood_list[l_index]:
        # print("from: " + str(l_index) + " -> " + str(n_index))

        line = [[X_dataset[l_index], Y_dataset[l_index]], [X_dataset[n_index], Y_dataset[n_index]]]
        pruned_graph_edges += [line]
pruned_graph_edges_collection = LineCollection(pruned_graph_edges, linewidths=(1), colors = ['r'], linestyle='solid')

# colors
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

# ax.set_xlim(0.0, 1.0)
# ax.set_ylim(0.0, 1.0)

if not args.dont_display_grid:
    ax.scatter(X_grid, Y_grid, c='y', zorder=3)

ax.scatter(X_dataset, Y_dataset, c='k', zorder=3)

cs = ax.contourf(X_density_grid, Y_density_grid, DensityValues, zorder=1,cmap=plt.cm.jet) #, zorder=2
fig.colorbar(cs)

if not args.dont_display_knn:
    if not args.use_unpruned_graph:
        ax.add_collection(pruned_graph_edges_collection)
    else:
        ax.add_collection(graph_edges_collection)

# plt.show()
if args.dont_display_knn:
    if args.dont_display_grid:
        fig.savefig("graphs/" + args.scenario_name + "_density.png", dpi=300)
    else:
        fig.savefig("graphs/" + args.scenario_name + "_density_with_grid.png", dpi=300)
else:
    if args.use_unpruned_graph:
        if args.dont_display_grid:
            fig.savefig("graphs/" + args.scenario_name + "_unpruned.png", dpi=300)
        else:
            fig.savefig("graphs/" + args.scenario_name + "_unpruned_with_grid.png", dpi=300)
    else:
        if args.dont_display_grid:
            fig.savefig("graphs/" + args.scenario_name + "_pruned.png", dpi=300)
        else:
            fig.savefig("graphs/" + args.scenario_name + "_pruned_with_grid.png", dpi=300)

plt.close()
