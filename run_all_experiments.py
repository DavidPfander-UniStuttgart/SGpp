#!/usr/bin/python3

import sys
import shlex
import subprocess
import re
from itertools import chain

if len(sys.argv) > 3 or len(sys.argv) < 3:
    print("error: config or device not provided (or too many arguments)")
    exit()

configFile = sys.argv[1]
print("configFile: " + configFile)

precision = "double"
if configFile.find("double") == -1:
    precision = "float"
print("precision: " + precision)

deviceName = sys.argv[1]
print("deviceName: " + deviceName)

# common dataset parameters
clusters=3
print("clusters: " + str(clusters))
abweichung = 0.05
print("abweichung: " + str(abweichung))
clusters_distance = 3 # required distance between cluster centers, criterion dis < c_dis * abw
print("clusters_distance: " + str(clusters_distance))
# noise = 0
# print("noise: " + )

# common
levels = {2: 8, 4: 7, 6: 6, 8: 5, 10: 4}
print("levels: " + str(levels))
lambdas = 1E-2
print("lambda: " + str(lambdas))
threshold = 0.2
print("threshold: " + str(threshold))
k =5
print("k: " + str(k))

# refinement
refinement_steps=1
refinement_points=0
coarsen_points=10000
coarsen_threshold=1E-3

CSV_SEP = ";"

# for dim in range(2, 3, 2):
for dim in range(2, 11, 2):

    f_result = open("results/results_gaussian_c" + str(clusters) + "_dim" + str(dim) + "_" + deviceName + "_" + precision + ".csv", "w")
    f_result.write("dataset_size" + CSV_SEP + "refinement_steps" + CSV_SEP + "total_duration_generate_b" + CSV_SEP + "avr_gflops_generate_b" + CSV_SEP + "total_duration_density" + CSV_SEP + "avr_gflops_density" + CSV_SEP + "duration_create_graph" + CSV_SEP + "gflops_create_graph" + CSV_SEP + "duration_prune_graph" + CSV_SEP + "gflops_prune_graph\n")

    # for dataset_size in [200]:
    for dataset_size in chain(range(20000, 110000, 20000), range(200000, 1100000, 200000)):
      dataset_file_name = "datasets/gaussian_c" + str(clusters) + "_size" + str(dataset_size) + "_dim" + str(dim) + ".arff"
      print("experiments for " + dataset_file_name)
      cmd = "./datadriven/examplesOCL/clustering_cmd --datasetFileName " + dataset_file_name + " --level " + str(levels[dim]) + " --lambda " + str(lambdas) + " --threshold " + str(threshold) + " --k " + str(k) + " --config " + configFile + " --refinement_steps " + str(refinement_steps) + " --refinement_points " + str(refinement_points) + " --coarsen_points " + str(coarsen_points) + " --coarsen_threshold " + str(coarsen_threshold)
      print("cmd: " + cmd)
      p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE)
      output = p.communicate()[0]
      output = output.decode('utf-8')
      print(output)

      # avr generate_b gflops
      count_generate_b = 0
      avr_gflops_generate_b = 0
      for g in re.finditer(r"flops_generate_b: (.*?) GFLOPS", output):
          avr_gflops_generate_b += float(g.group(1))
          count_generate_b += 1
      avr_gflops_generate_b /= float(count_generate_b)

      # total generate_b duration
      total_duration_generate_b = 0
      for g in re.finditer(r"last_duration_generate_b: (.*?)s", output):
          total_duration_generate_b += float(g.group(1))

      # avr density gflops
      count_density = 0
      avr_gflops_density = 0
      for g in re.finditer(r"flops_density: (.*?) GFLOPS", output):
          avr_gflops_density += float(g.group(1))
          count_density += 1
      avr_gflops_density /= float(count_density)

      # total density duration
      total_duration_density = 0
      for g in re.finditer(r"acc_duration_density: (.*?)s", output):
          total_duration_density += float(g.group(1))

      g = re.search(r"flops_create_graph: (.*?) GFLOPS", output)
      gflops_create_graph = g.group(1)

      g = re.search(r"last_duration_create_graph: (.*?)s", output)
      duration_create_graph = g.group(1)

      g = re.search(r"flops_prune_graph: (.*?) GFLOPS", output)
      gflops_prune_graph = g.group(1)

      g = re.search(r"last_duration_prune_graph: (.*?)s", output)
      duration_prune_graph = g.group(1)

      f_result.write(str(dataset_size) + CSV_SEP + str(refinement_steps) + CSV_SEP + str(total_duration_generate_b) + CSV_SEP + str(avr_gflops_generate_b) + CSV_SEP + str(total_duration_density) + CSV_SEP + str(avr_gflops_density) + CSV_SEP + str(duration_create_graph) + CSV_SEP + str(gflops_create_graph) + CSV_SEP + str(duration_prune_graph) + CSV_SEP + str(gflops_prune_graph) + "\n")
