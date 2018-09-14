#!/usr/bin/python3

import subprocess, shlex

# master node is added automatically
nodes_start = 1
nodes_end = 16
config_file = "config_ocl_float_P100.cfg"
dataset_size = 1000000
dim = 10
dataset_file = "$SCRATCH/datasets/gaussian_c3_size" + str(dataset_size) + "_dim" + str(dim) + ".arff"
level = 6
lambda_value = 1E-6
threshold = 0.7

for nodes in range(nodes_start, nodes_end):
    cmd = shlex.split("srun -N " + str(nodes + 1) + " -C gpumodedefault --time 00:10:00 ./datadriven/examplesOCL/clustering_cmd --config=" + config_file + " --threshold=" + str(threshold) + " --datasetFileName=" + dataset_file + "  --level=" + str(level) + " --lambda=" + str(lambda_value) + " > piz_daint_" + str(nodes) + "N.log 2>&1")
    print(cmd)
