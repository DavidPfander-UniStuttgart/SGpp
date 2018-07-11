#!/usr/bin/python

import random
import numpy as np
from itertools import chain
import argparse

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

def write_csv_without_erg(f, n_features, points):
    for i in range(len(points)):
        for d in range(n_features-1):
            f.write(str(points[i][d]) + str(", "))
        f.write(str(points[i][n_features-1]) + str("\n"))

def write_erg(f, classes):
    for i in range(len(classes)):
        f.write(str(classes[i]) + str("\n"))

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
         for i in range(0, min(currentsize, setsize // clusters)):
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

if __name__ == '__main__':
    # dimensions, clusters, setsize, abweichung, rauschensize
    parser = argparse.ArgumentParser(description='Generate clusterings datasets using gauss blobs.')
    parser.add_argument('-c', '--clusters', type=int, required=True,
                    help='Number of clusters')
    parser.add_argument('-dev', '--deviation', type=float, default=0.12,
                        help='Standard deviation of the clusters (default 0.12). Will be \
                        automatically reduced if the clusters are not well seperated. See \
                        distance argument')
    parser.add_argument('-dis', '--distance', type=int, required=True,
                        help='Enforced distance between clusters (measured in the standard deviation of the gauss blobs)')
    parser.add_argument('-dim', '--dimensions', type=int, required=True,
                    help='Number of dimensions')
    parser.add_argument('-s', '--size', type=int, required=True,
                    help='Number of datapoints')
    parser.add_argument('-n', '--noise_size', type=int, default=0,
                        help='Number of noise datapoints (will be added on top of original datasize)')
    parser.add_argument('-f', '--filename', type=str, required=True,
                    help='Filename (with path) for generated dataset (arff format)')
    parser.add_argument('-fc', '--erg_filename', type=str, required=True,
                    help='Filename (with path) for generated dataset (arff format)')
    args = parser.parse_args()
    print("Number of clusters: ", args.clusters)
    print("Dimension of generated dataset: ", args.dimensions)
    print("Number of data points in generated dataset: ", args.size)
    print("Number of noise datapoints: ", args.noise_size)
    print("Filename of dataset (arff): ", args.filename)
    print("Filename of expected results clusters: ", args.erg_filename)

    print("Generating dataset", args.filename, " ... ")
    random.seed(0) # Always generate the same datasets
    dataset1, Y1, centers = generate_dataset(args.dimensions, args.clusters, args.size, args.deviation, args.noise_size, args.distance)
    f = open(args.filename, "w")
    write_arff_header(f, args.dimensions, args.filename, False)
    write_csv_without_erg(f, args.dimensions, dataset1)
    f = open(args.erg_filename, "w")
    write_erg(f, Y1)
    print("Finished!")
