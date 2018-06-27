#!/usr/bin/python

import random
import numpy as np
from itertools import chain

def write_arff_header(f, n_features, name, write_class=False):
    f.write("@RELATION \"" + name + "\"\n\n")
    for i in range(n_features):
        f.write("@ATTRIBUTE x" + str(i) + " NUMERIC\n")
    if write_class:
        f.write("@ATTRIBUTE class NUMERIC\n")
    f.write("\n@DATA\n")

def write_csv(f, n_features, points, classes):
    for i in range(len(points)):
        for d in range(n_features):
            f.write(str(points[i][d]) + str(", "))
        f.write(str(classes[i]) + "\n")

def normiere_liste(liste, dimensions):
   B = np.array(liste)
   mins = []
   maxs = []
   for dimension in range(0, dimensions):
      teil = B[:, dimension]
      minimum = min(teil)
      maximum = max(teil)
      mins += [minimum]
      maxs += [maximum]
      for i in range(0, len(teil)):
         teil[i] = (teil[i]-minimum)/(maximum-minimum)
         # teil[i] = (teil[i]+0.1)*0.8
      B[:, dimension] = teil
   return B, mins, maxs

def normiere_centers(liste, dimensions, mins, maxs):
   B = np.array(liste)
   for dimension in range(0, dimensions):
      teil = B[:, dimension]
      # minimum = min(teil)
      # maximum = max(teil)
      minimum = mins[dimension]
      maximum = maxs[dimension]
      for i in range(0, len(teil)):
         teil[i] = (teil[i]-minimum)/(maximum-minimum)
         # teil[i] = (teil[i]+0.1)*0.8
      B[:, dimension] = teil
   return B

def generate_dataset(dimensions, clusters, setsize, abweichung, rauschensize, clusters_distance):
   #generate centers
   centers = []
   for cluster in range(0, clusters):
      abstand = 0
      center = []
      counter = 0
      continue_search = 1
      while continue_search:
         center[:] = []
         if counter == 50:
            # print "in here"
            abweichung = abweichung * 0.9
            counter = 0
         for dimension in range(0, dimensions):
            center.append(random.random())
         #check point
         continue_search = 0
         for c in centers:
            abstand = 0
            for dimension in range(0, dimensions):
               abstand = abstand + (center[dimension] - c[dimension])**2
            abstand = abstand ** 0.5
            if abstand < clusters_distance * abweichung:
               continue_search = 1
               break
         counter = counter +1
         if not centers:
            break
      centers.append(center)
   #generate datapoints
   dataset = []
   cluster_ret = []
   currentsize = setsize
   for cluster in range(0, clusters):
      if currentsize > 0:
         for i in range(0, min(currentsize, setsize / clusters)):
            point = []
            for dimension in range(0, dimensions):
               point.append(random.gauss(centers[cluster][dimension], abweichung))
            dataset.append(point)
            cluster_ret.append(cluster+1)
            currentsize = currentsize - 1
         #currentsize = currentsize - setsize / clusters
   while currentsize is not 0:
      point = []
      for dimension in range(0, dimensions):
         point.append(random.gauss(centers[cluster][dimension], abweichung))
      dataset.append(point)
      currentsize = currentsize - 1
      cluster_ret.append(clusters)
   for i in range(0, rauschensize):
      point = []
      for dimension in range(0, dimensions):
         # point.append(random.uniform(0.05,0.95))
        point.append(random.uniform(0.0,1.0))
      dataset.append(point)
      cluster_ret.append(-1)
   dataset, mins, maxs = normiere_liste(dataset, dimensions)
   centers = normiere_centers(centers, dimensions, mins, maxs)
   cluster_npret = np.array(cluster_ret)
   # print dataset
   return dataset, cluster_npret, centers

# dimensions, clusters, setsize, abweichung, rauschensize
clusters=3
abweichung = 0.05
clusters_distance = 3 # required distance between cluster centers, criterion dis < c_dis * abw
noise_percent = 0.1
# for dim in range(2, 11, 2):
for dim in range(2, 3, 2):
    for dataset_size in chain([200], range(20000, 110000, 20000), range(200000, 1100000, 200000)):
      file_name = "datasets/gaussian_c" + str(clusters) + "_size" + str(dataset_size) + "_dim" + str(dim)
      print "creating " + file_name + ".arff"
      noise = int(noise_percent * dataset_size)
      dataset1, Y1, centers = generate_dataset(dim, clusters, dataset_size, abweichung, noise, clusters_distance)

      f = open(file_name + ".arff", "w")
      write_arff_header(f, dim, file_name, True)
      write_csv(f, dim, dataset1, Y1)

      f = open(file_name + "_centers.arff", "w")
      write_arff_header(f, dim, file_name)
      for center in centers:
          for d in range(dim):
              if (d > 0):
                  f.write(", ")
              f.write(str(center[d]))
          f.write("\n")
