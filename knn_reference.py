#!/usr/bin/python3

from sklearn.neighbors import NearestNeighbors
import numpy as np
import os.path

datasets_folder = 'datasets/'
extension = '.arff'
csv_separator = ','
k = 5

# find all datasets to process
data_files = []
for entry in os.scandir(datasets_folder):
    print(entry.name)
    if entry.is_dir():
        pass
    elif entry.is_file():
        if entry.name.endswith(extension) and not entry.name.endswith('_centers' + extension):
            data_files += [os.path.join(datasets_folder, entry.name)]

print('all datasets:', data_files)
# data_files = [datasets_folder + 'gaussian_c3_size200_dim2.arff']

for csv_file_name in data_files:
    print('next dataset:', csv_file_name)
    # csv_file_name = 'datasets/gaussian_c3_size200_dim2.arff'
    csv_file_basename = os.path.basename(csv_file_name)
    indices_file_name = os.path.join(datasets_folder, os.path.splitext(csv_file_basename)[0] + '_indices.csv')
    # distances_file_name = os.path.join(datasets_folder, os.path.basename(csv_file_name) + '_distance.csv')
    print('indices_file_name:', indices_file_name)


    f = open(csv_file_name, "rb")
    skip_counter = 0
    for r in f:
        skip_counter += 1
        if r == b'@DATA\n':
            break

    # dim = 2
    has_class = True
    # reader = np.loadtxt(open(csv_file_name, "rb"), delimiter=",", skiprows=dim + 5)
    reader = np.loadtxt(open(csv_file_name, "rb"), delimiter=csv_separator, skiprows=skip_counter)

    X = list(reader)
    dim = len(X[0])
    if has_class:
        dim -= 1
    X = np.array(X).astype("float")
    X = X[:,0:dim]
    print(X)

    # X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    indices = indices[:,1:k+1].astype(int)
    # distances = distances[:,1:k+1]

    np.savetxt(indices_file_name, indices, delimiter=csv_separator, fmt="%u")
    # np.savetxt(distances_file_name, indices, delimiter=csv_separator)
