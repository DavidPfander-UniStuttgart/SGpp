#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import csv
import argparse

def read_arff(file_name):
    # read-in dataset
    f = open(file_name, 'r')
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
    return X_dataset, Y_dataset

parser = argparse.ArgumentParser(description='Create graphs.')
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument('--dataset_name', required=True)
args = parser.parse_args()

# # read-in dataset
X_dataset, Y_dataset = read_arff(args.dataset_name)
center_file = args.dataset_name.replace(".arff", "_centers.arff")
X_centers, Y_centers = read_arff(center_file)

##################################################
# Estimate density
##################################################

fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)

# ax.set_xlim(0.0, 1.0)
# ax.set_ylim(0.0, 1.0)

ax.scatter(X_dataset, Y_dataset, c='k') # , zorder=3

print(X_centers)
print(Y_centers)

ax.scatter(X_centers, Y_centers, c='r') # , zorder=3

plt.show()
plt.close()
