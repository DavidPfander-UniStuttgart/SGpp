#!/usr/bin/python

#
# Reads in a level index tuple containing csv file and plots it.
#

import csv
import matplotlib.pyplot as plt

dim = 2
filename = "grid.csv"
filename_dataset = "dataset2_dim2.arff"
filename_dataset_skip_rows = 10

csvfile = open(filename, "r")
csvreader = csv.reader(csvfile, delimiter=',', quotechar='#')

def li_to_coords(l, i):
    coords = []
    for d in range(dim):
        h = 2**-l[d]
        coords += [h * i[d]]
    return coords


fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)

print "processing data"
    
csvfile_data = open(filename_dataset, "r")
csvreader_data = csv.reader(csvfile_data, delimiter=',', quotechar='#')
color_data="b"
row_counter = 0
for row in csvreader_data:
    # print row
    row_counter +=1 
    if row_counter <= filename_dataset_skip_rows:
        continue

    if not row_counter % 10 == 0:
        continue

    point = []
    for i in range(dim):
        point += [float(row[i])]

    # print point

    ax.scatter(point[0], point[1], c=color_data)

print "data processed, processing grid"

color_grid="r"

row_counter = 0
for row in csvreader:
    # print row
    row_counter +=1 
    if row_counter == 1:
        continue

    level = []
    for i in range(dim):
        level += [float(row[i])]
    index = []
    for i in range(dim):
        index += [float(row[dim + i])]
    # print level, index
    coords = li_to_coords(level, index)
    # print coords

    ax.scatter(coords[0], coords[1], c=color_grid)    

plt.show()
plt.close()
