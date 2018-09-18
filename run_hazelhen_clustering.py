#!/usr/bin/python3

import subprocess, shlex

# dataset_sizes = [10000000, 100000000]
# dataset_sizes = [10000, 100000]
dataset_sizes=[1000000]

# master node is added automatically
config_file="config_ocl_float_hazelhen.cfg"
# config_file = "config_ocl_float_XeonE31585v5.cfg"
dim = 10
# dataset_file = "datasets/gaussian_c3_size" + str(dataset_size) + "_dim" + str(dim) + ".arff"
level = 3
lambda_value = 1E-6
threshold = 0.7
epsilon = 1E-3

nodes_start=1
nodes_end=4

for dataset_size in dataset_sizes:
    nodes = nodes_start
    # for nodes in range(nodes_start, nodes_end):
    while nodes <= nodes_end:
        # cmd_raw = "qsub -n " + str(nodes) + " hazelhen_clustering.job " + str(dataset_size) + " " + str(config_file) + " " + str(dim) + " " + str(level) + " " + str(lambda_value) + " " + str(threshold) + " " + str(epsilon)
        cmd_raw = "qsub -l nodes=" + str(nodes + 1) + ":ppn=1,walltime=00:02:00 hazelhen_clustering.job -F \"" + str(dataset_size) + " " + str(config_file) + " " + str(dim) + " " + str(level) + " " + str(lambda_value) + " " + str(threshold) + " " + str(epsilon) + "\""
        print(cmd_raw)
        cmd = shlex.split(cmd_raw)
        p = subprocess.Popen(cmd)
        p.communicate()
        nodes *= 2
