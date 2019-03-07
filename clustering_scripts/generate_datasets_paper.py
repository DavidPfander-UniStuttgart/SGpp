#!/usr/bin/python3

import random
import numpy as np
from itertools import chain
import math

import argparse

parser = argparse.ArgumentParser(description='Create datasets.')
# parser.add_argument('--eval_level', dest='eval_level', action='store_const')
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument('--output_folder', type=str, required=True)

domain_min = 0.1
domain_max = 0.9

def write_arff_header(f, n_features, name, write_class=False):
    f.write("@RELATION \"" + name + "\"\n\n")
    for i in range(n_features):
        f.write("@ATTRIBUTE x" + str(i) + " NUMERIC\n")
    if write_class:
        f.write("@ATTRIBUTE class NUMERIC\n")
    f.write("\n@DATA\n")

def write_csv(f, n_features, points):
    for i in range(len(points)):
        for d in range(n_features):
            if (d > 0):
                f.write(", ")
            f.write(str(points[i][d]))
        f.write("\n")

def write_all_arffs(dim, file_name, dataset1, Y1, centers):
    f = open(file_name + ".arff", "w")
    write_arff_header(f, dim, file_name, False)
    write_csv(f, dim, dataset1)

    f = open(file_name + "_class.arff", "w")
    for i in range(len(Y1)):
        f.write(str(Y1[i]) + "\n")
    f.close()

    f = open(file_name + "_centers.arff", "w")
    # write_arff_header(f, dim, file_name)
    for center in centers:
        for d in range(dim):
            if (d > 0):
                f.write(", ")
            f.write(str(center[d]))
        f.write("\n")
    f.close()

def normalize(dataset, centers, dimensions):
    # B = np.array(liste)
    mins = []
    maxs = []
    for dimension in range(0, dimensions):
        col = dataset[:, dimension]
        minimum = min(col)
        maximum = max(col)
        mins += [minimum]
        maxs += [maximum]
        if maximum == minimum and minimum >= 0.0 and minimum <= 1.0:
            continue
        for i in range(0, len(col)):
            col[i] = (col[i] - minimum)/(maximum - minimum)
            col[i] = col[i] * (domain_max - domain_min) + domain_min
            # col[i] = (col[i]+0.1)*0.8
        dataset[:, dimension] = col

    for dimension in range(0, dimensions):
        col = centers[:, dimension]
        minimum = mins[dimension]
        maximum = maxs[dimension]
        if maximum == minimum and minimum >= 0.0 and minimum <= 1.0:
            continue
        for i in range(0, len(col)):
            col[i] = (col[i] - minimum)/(maximum - minimum)
            col[i] = col[i] * (domain_max - domain_min) + domain_min
            # col[i] = (col[i]+0.1)*0.8
        centers[:, dimension] = col

def generate_dataset(dimensions, num_clusters, setsize, deviation, clusters_distance, cutoff_radius, additional_dims):
    #generate centers
    centers = np.zeros(shape=(num_clusters, dimensions + additional_dims))
    for cluster in range(0, num_clusters):
        distance = 0
        center = []
        counter = 0
        continue_search = 1
        while continue_search:
            center[:] = []
            if counter == 100:
                print("error: could not generate cluster centers")
                raise SystemExit
            for dimension in range(0, dimensions):
                center.append(random.uniform(0.0, 1.0))
            for dimension in range(0, additional_dims):
                center.append(0.5)
            #check point
            continue_search = 0
            for c in centers:
                distance = 0
                for dimension in range(0, dimensions + additional_dims):
                    distance = distance + (center[dimension] - c[dimension])**2
                distance = math.sqrt(distance) # ** 0.5
                if distance < clusters_distance * deviation:
                    continue_search = 1
                    break
            counter = counter +1
        centers[cluster] = center
    print("centers:")
    for c in centers:
        print(c)

    #generate datapoints
    dataset = np.zeros(shape=(setsize, dimensions + additional_dims))
    cluster_ret = np.zeros(shape=(setsize), dtype=int)
    currentsize = 0
    cluster_size = setsize / num_clusters
    print("create clusters")
    for cluster in range(0, num_clusters):
        if currentsize >= setsize:
            print("error: cluster was not generated!")
            raise SystemExit
        # if currentsize < setsize:
        # for i in range(0, min(currentsize, setsize / num_clusters)):
        i = 0
        while i < cluster_size:
        # for i in range(0, cluster_size):
            if dataset_type == "gaussian":
                for dimension in range(0, dimensions):
                    dataset[currentsize, dimension] = random.gauss(centers[cluster][dimension], deviation)
            elif dataset_type == "hypercube":
                for dimension in range(0, dimensions):
                    dataset[currentsize, dimension] = random.uniform(centers[cluster][dimension] - deviation, centers[cluster][dimension] + deviation)

            for dimension in range(0, additional_dims):
                dataset[currentsize, dimensions + dimension] = 0.5

            dist = 0.0
            for dimension in range(0, dimensions + additional_dims):
                temp = dataset[currentsize, dimension] - centers[cluster][dimension]
                dist += temp * temp
            dist = math.sqrt(dist)
            if dist > cutoff_radius * deviation:
                continue
            cluster_ret[currentsize] = cluster + 1
            currentsize = currentsize + 1
            if currentsize >= setsize:
                break
            i += 1

    # cluster size does not divide setsize, fill up last cluster
    print("fillup currentsize:", currentsize, "setsize:", setsize)
    while currentsize < setsize:
        if dataset_type == "gaussian":
            for dimension in range(0, dimensions):
                dataset[currentsize, dimension] = random.gauss(centers[cluster][dimension], deviation)
        elif dataset_type == "hypercube":
            for dimension in range(0, dimensions):
                dataset[currentsize, dimension] = random.uniform(centers[cluster][dimension] - deviation, centers[cluster][dimension] + deviation)

        for dimension in range(0, additional_dims):
            dataset[currentsize, dimensions + dimension] = 0.5

        dist = 0.0
        for dimension in range(0, dimensions + additional_dims):
            temp = dataset[currentsize, dimension] - centers[cluster][dimension]
            dist += temp * temp
        dist = math.sqrt(dist)
        if dist > cutoff_radius * deviation:
            continue

        cluster_ret[currentsize] = num_clusters
        currentsize = currentsize + 1

    print("created, now normalizing")

    normalize(dataset, centers, dimensions + additional_dims)
    # for dimension in range(0, dimensions):
    #     col = dataset[:, dimension]
    #     minimum = min(col)
    #     maximum = max(col)
    #     print("minimum:", minimum, "maximum:", maximum)

    print("normalized, now randomizing")

    # for i in [0]:
    for i in range(0, setsize):
        swap_index = random.randint(0, setsize - 1)
        temp = dataset[i].copy()
        dataset[i] = dataset[swap_index]
        dataset[swap_index] = temp
        temp = cluster_ret[i].copy()
        cluster_ret[i] = cluster_ret[swap_index]
        cluster_ret[swap_index] = temp

    return dataset, cluster_ret, centers

def add_noise(dimensions, setsize, num_noise, dataset1, Y1, additional_dims):
    # dataset = dataset1.copy()
    noise_data = np.zeros(shape=(num_noise, dimensions + additional_dims))
    cluster_noise = np.zeros(shape=(num_noise), dtype=int)
    print("create rauschen")
    for i in range(0, num_noise):
        for dimension in range(0, dimensions):
            noise_data[i, dimension] = random.uniform(domain_min, domain_max)
        for dimension in range(dimensions, dimensions + additional_dims):
            noise_data[i, dimension] = 0.5
        cluster_noise[i] = -1

    dataset = np.concatenate((dataset1, noise_data), axis = 0)
    cluster_ret = np.concatenate((Y1, cluster_noise), axis = 0)

    for i in range(0, setsize + num_noise):
        swap_index = random.randint(0, setsize + num_noise - 1)
        temp = dataset[i].copy()
        dataset[i] = dataset[swap_index]
        dataset[swap_index] = temp
        temp = cluster_ret[i].copy()
        cluster_ret[i] = cluster_ret[swap_index]
        cluster_ret[swap_index] = temp

    return dataset, cluster_ret

# dimensions, clusters, setsize, deviation, noise_percent
np.set_printoptions(precision=3)
num_clusters = 4
deviation = 0.05
clusters_distance = 7 # unit: standard deviation
cutoff_radius = 3 # unit: standard deviation
noise_percent = 0.05
datasets_folder='datasets_diss/'
additional_dims = 0
# dataset_type = "gaussian" # 'gaussian' or 'hypercube'
for dim in [2]: # [5, 10]
    for dataset_size in [int(500)]: # [1000000, 10000000, 100000000]
        for dataset_type in ["gaussian"]: # , "hypercube"
            file_name = datasets_folder + dataset_type + "_c" + str(num_clusters) + "_size" + str(dataset_size) + "_dim" + str(dim + additional_dims) + "id" + str(dim)
            print("creating " + file_name + ".arff")
            # always create dataset without noise first
            dataset1, Y1, centers = generate_dataset(dim, num_clusters, dataset_size, deviation, clusters_distance, cutoff_radius, additional_dims)
            write_all_arffs(dim + additional_dims, file_name, dataset1, Y1, centers)
            num_noise = int(noise_percent * dataset_size)
            assert((num_noise > 0) or (noise_percent == 0))
            if num_noise > 0:
                file_name += "_noise"
                print("creating " + file_name + ".arff")
                dataset1_noise, Y1_noise = add_noise(dim, dataset_size, num_noise, dataset1, Y1, additional_dims)
                write_all_arffs(dim + additional_dims, file_name, dataset1_noise, Y1_noise, centers)
