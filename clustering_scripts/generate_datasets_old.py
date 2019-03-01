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
            # print(col[i])
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


# def normalize(dataset, centers, dimensions):
#    # B = np.array(liste)
#    mins = []
#    maxs = []
#    for dimension in range(0, dimensions):
#       teil = dataset[:, dimension]
#       minimum = min(teil)
#       maximum = max(teil)
#       mins += [minimum]
#       maxs += [maximum]
#       for i in range(0, len(teil)):
#          teil[i] = (teil[i] - minimum)/(maximum - minimum)
#          teil[i] = teil[i] * (domain_max - domain_min) + domain_min
#          # teil[i] = (teil[i]+0.1)*0.8
#       dataset[:, dimension] = teil

#    for dimension in range(0, dimensions):
#       teil = centers[:, dimension]
#       minimum = mins[dimension]
#       maximum = maxs[dimension]
#       for i in range(0, len(teil)):
#          teil[i] = (teil[i] - minimum)/(maximum - minimum)
#          teil[i] = teil[i] * (domain_max - domain_min) + domain_min
#          # teil[i] = (teil[i]+0.1)*0.8
#       centers[:, dimension] = teil

def generate_dataset(dimensions, num_clusters, setsize, abweichung, rauschensize, clusters_distance, additional_dims, additional_dims_random):
   #generate centers
   centers = np.zeros(shape=(num_clusters, dimensions + additional_dims))
   for cluster in range(0, num_clusters):
      abstand = 0
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
         #check point
         continue_search = 0
         for c in centers:
            abstand = 0
            for dimension in range(0, dimensions):
               abstand = abstand + (center[dimension] - c[dimension])**2
            abstand = math.sqrt(abstand) # ** 0.5
            if abstand < clusters_distance * abweichung:
               continue_search = 1
               break
         counter = counter +1
         for d in range(additional_dims):
             if additional_dims_random:
                 pass
             else:
                 center.append(0.5)
      centers[cluster] = center

   #generate datapoints
   dataset = np.zeros(shape=(setsize + rauschensize, dimensions + additional_dims))
   cluster_ret = np.zeros(shape=(setsize + rauschensize), dtype=int)
   currentsize = 0
   cluster_size = int(setsize / num_clusters)
   print("create clusters, cluster_size:", cluster_size)
   for cluster in range(0, num_clusters):
       if currentsize >= setsize:
            print("error: cluster was not generated!")
            raise SystemExit
      # if currentsize < setsize:
         # for i in range(0, min(currentsize, setsize / num_clusters)):
       for i in range(0, cluster_size):
            for dimension in range(0, dimensions):
                dataset[currentsize, dimension] = random.gauss(centers[cluster][dimension], abweichung)
            for d in range(additional_dims):
                if additional_dims_random:
                    pass
                else:
                    dataset[currentsize, dimensions + d] = 0.5
            cluster_ret[currentsize] = cluster + 1
            currentsize = currentsize + 1
            if currentsize >= setsize:
                break

   # cluster size does not divide setsize, fill up last cluster
   while currentsize < setsize:
      for dimension in range(0, dimensions):
          dataset[currentsize, dimension] = random.gauss(centers[cluster][dimension], abweichung)
      cluster_ret[currentsize] = num_clusters
      currentsize = currentsize + 1

   print("create rauschen")
   for i in range(0, rauschensize):
      for dimension in range(0, dimensions + additional_dims):
          dataset[setsize + i, dimension] = random.uniform(0.0, 1.0)
      cluster_ret[setsize + i] = -1

   print("created, now normalizing")

   normalize(dataset, centers, dimensions + additional_dims)
   # for dimension in range(0, dimensions):
   #    col = dataset[:, dimension]
   #    minimum = min(col)
   #    maximum = max(col)
   #    print("minimum:", minimum, "maximum:", maximum)

   print("normalized, now randomizing")

   for i in range(setsize + rauschensize):
       swap_index = random.randint(0, setsize + rauschensize - 1)
       temp = dataset[i]
       dataset[i] = dataset[swap_index]
       dataset[swap_index] = temp
       temp = cluster_ret[i]
       cluster_ret[i] = cluster_ret[swap_index]
       cluster_ret[swap_index] = temp

   return dataset, cluster_ret, centers

# dimensions, clusters, setsize, abweichung, rauschensize
num_clusters=10
abweichung = 0.05
clusters_distance = 6 # required distance between cluster centers, criterion dis < c_dis * abw
additional_dims = 6
additional_dims_random = False
# noise_percent = 0.02
noise_dims = 5
# for dim in range(2, 11, 2):
for dim in range(4, 5, 2):
    for noise_percent in [0.0, 0.02]:
        # for dataset_size in [10000000, 100000000]:
        for dataset_size in [1000000]:
        # for dataset_size in [100]:
            # for dataset_size in chain([200], range(20000, 110000, 20000), range(200000, 1100000, 200000)):
          file_name = "datasets_diss/gaussian_c" + str(num_clusters) + "_size" + str(dataset_size) + "_dim" + str(dim + additional_dims) + "id" + str(dim)
          if noise_percent > 0.0:
              file_name += "_noise"
          print("creating " + file_name + ".arff")
          noise_size = int(noise_percent * dataset_size)
          dataset1, Y1, centers = generate_dataset(dim, num_clusters, dataset_size, abweichung, noise_size, clusters_distance, additional_dims, additional_dims_random)

          f = open(file_name + ".arff", "w")
          write_arff_header(f, dim + additional_dims, file_name, False)
          write_csv(f, dim + additional_dims, dataset1)

          f = open(file_name + "_class.arff", "w")
          for i in range(len(Y1)):
              f.write(str(Y1[i]) + "\n")
          f.close()

          f = open(file_name + "_centers.arff", "w")
          # write_arff_header(f, dim, file_name)
          for center in centers:
              for d in range(dim + additional_dims):
                  if (d > 0):
                      f.write(", ")
                  f.write(str(center[d]))
              f.write("\n")
          f.close()
