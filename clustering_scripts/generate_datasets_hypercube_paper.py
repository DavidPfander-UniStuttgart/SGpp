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
      teil = dataset[:, dimension]
      minimum = min(teil)
      maximum = max(teil)
      mins += [minimum]
      maxs += [maximum]
      for i in range(0, len(teil)):
         teil[i] = (teil[i] - minimum)/(maximum - minimum)
         teil[i] = teil[i] * (domain_max - domain_min) + domain_min
         # teil[i] = (teil[i]+0.1)*0.8
      dataset[:, dimension] = teil

   for dimension in range(0, dimensions):
      teil = centers[:, dimension]
      minimum = mins[dimension]
      maximum = maxs[dimension]
      for i in range(0, len(teil)):
         teil[i] = (teil[i] - minimum)/(maximum - minimum)
         teil[i] = teil[i] * (domain_max - domain_min) + domain_min
         # teil[i] = (teil[i]+0.1)*0.8
      centers[:, dimension] = teil

def generate_dataset(dimensions, num_clusters, dataset_size, deviation, rauschensize, clusters_distance):
   #generate centers
   centers = np.zeros(shape=(num_clusters, dimensions))
   for cluster in range(0, num_clusters):
      distance = 0
      center = []
      counter = 0
      continue_search = 1
      while continue_search:
         center[:] = []
         if counter == 500:
             print("error: could not generate cluster centers")
             raise SystemExit
         for dimension in range(0, dimensions):
             center.append(random.uniform(0.0, 1.0))
         #check point
         continue_search = 0

         # # check distance from center
         # distance_center = 0.0
         # for dimension in range(0, dimensions):
         #     distance_center += (center[dimension] - 0.5)**2
         # distance_center = math.sqrt(distance_center) # ** 0.5
         # # if (distance_center > 1.2):
         # #     print(distance_center)
         # if distance_center > 1.0:
         #     continue_search = 1
         #     counter += 1
         #     continue
         for c in centers:
            distance = 0.0
            for dimension in range(0, dimensions):
               distance += (center[dimension] - c[dimension])**2
            distance = math.sqrt(distance) # ** 0.5
            # print("distance:", distance)
            if distance < clusters_distance:
               continue_search = 1
               break
         counter = counter +1
      centers[cluster] = center

   #generate datapoints
   dataset = np.zeros(shape=(dataset_size + rauschensize, dimensions))
   cluster_ret = np.zeros(shape=(dataset_size + rauschensize), dtype=int)
   currentsize = 0
   cluster_size = dataset_size / num_clusters
   print("create clusters")
   for cluster in range(0, num_clusters):
       if currentsize >= dataset_size:
            print("error: cluster was not generated!")
            raise SystemExit
      # if currentsize < dataset_size:
         # for i in range(0, min(currentsize, dataset_size / num_clusters)):
       for i in range(0, cluster_size):
            for dimension in range(0, dimensions):
                dataset[currentsize, dimension] = random.uniform(centers[cluster][dimension] - deviation, centers[cluster][dimension] + deviation)
                # if dimension == 0:
                #     print("center dim:", centers[cluster][dimension], "gen:", dataset[currentsize, dimension], "abw:", deviation)

            cluster_ret[currentsize] = cluster + 1
            currentsize = currentsize + 1
            if currentsize >= dataset_size:
                break

   # cluster size does not divide dataset_size, fill up last cluster
   while currentsize < dataset_size:
      for dimension in range(0, dimensions):
          dataset[currentsize, dimension] = random.uniform(centers[cluster][dimension] - deviation, centers[cluster][dimension] + deviation)
      cluster_ret[currentsize] = num_clusters
      currentsize = currentsize + 1
      
   print("create rauschen")
   for i in range(0, rauschensize):
      for dimension in range(0, dimensions):
          dataset[dataset_size + i, dimension] = random.uniform(0.0, 1.0)
      cluster_ret[dataset_size + i] = -1

   print("created, now normalizing")

   normalize(dataset, centers, dimensions)
   # for dimension in range(0, dimensions):
   #    col = dataset[:, dimension]
   #    minimum = min(col)
   #    maximum = max(col)
   #    print("minimum:", minimum, "maximum:", maximum)

   print("normalized, now randomizing")

   for i in range(dataset_size + rauschensize):
       swap_index = random.randint(0, dataset_size + rauschensize - 1)
       temp = dataset[i]
       dataset[i] = dataset[swap_index]
       dataset[swap_index] = temp
       temp = cluster_ret[i]
       cluster_ret[i] = cluster_ret[swap_index]
       cluster_ret[swap_index] = temp

   return dataset, cluster_ret, centers

# dimensions, clusters, dataset_size, deviation, rauschensize
num_clusters=5
deviation = 0.00001
clusters_distance = math.sqrt(10) * deviation * 10000 # required distance between cluster centers, criterion dis < c_dis * abw
print(clusters_distance)
# noise_percent = 0.02
# for dim in range(2, 11, 2):
for dim in range(10, 11, 2):
    for noise_percent in [0.0, 0.02]:
        for dataset_size in [100000]:
        # for dataset_size in [100]:
            # for dataset_size in chain([200], range(20000, 110000, 20000), range(200000, 1100000, 200000)):
          file_name = "final_paper_datasets/hypercube_c" + str(num_clusters) + "_size" + str(dataset_size) + "_dim" + str(dim)
          if noise_percent > 0.0:
              file_name += "_noise"
          print("creating " + file_name + ".arff")
          noise = int(noise_percent * dataset_size)
          dataset1, Y1, centers = generate_dataset(dim, num_clusters, dataset_size, deviation, noise, clusters_distance)

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
