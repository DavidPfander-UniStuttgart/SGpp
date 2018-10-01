#!/usr/bin/python3

import subprocess, shlex

# dataset_sizes = [10000000, 100000000]
# dataset_sizes = [10000, 100000]
dataset_sizes=[100000000]

# master node is added automatically
config_file="config_ocl_float_hazelhen.cfg"
# config_file = "config_ocl_float_XeonE31585v5.cfg"
dim = 10
# dataset_file = "datasets/gaussian_c3_size" + str(dataset_size) + "_dim" + str(dim) + ".arff"
level = 8
lambda_value = 1E-6
threshold = 0.7
epsilon = 1E-3

nodes_start= 16
nodes_end=16


for dataset_size in dataset_sizes:
    nodes = nodes_start
    # for nodes in range(nodes_start, nodes_end):

    # originally set to 25000 for l8 and 32 nodes
    seconds_total_smallest_no_of_nodes = 50000 # 1h
    
    while nodes <= nodes_end:
        minutes_total, seconds = divmod(seconds_total_smallest_no_of_nodes, 60)
        hours, minutes = divmod(minutes_total, 60)
        walltime_formatted = '{:02d}:{:02d}:{:02d}'.format(hours, minutes, seconds)
        # cmd_raw = "qsub -n " + str(nodes) + " hazelhen_clustering.job " + str(dataset_size) + " " + str(config_file) + " " + str(dim) + " " + str(level) + " " + str(lambda_value) + " " + str(threshold) + " " + str(epsilon)
        cmd_raw = "qsub -l nodes=" + str(nodes + 1) + ":ppn=1,walltime=" + walltime_formatted + " hazelhen_clustering.job -F \"" + str(dataset_size) + " " + str(config_file) + " " + str(dim) + " " + str(level) + " " + str(lambda_value) + " " + str(threshold) + " " + str(epsilon) + "\""
        print(cmd_raw)
        cmd = shlex.split(cmd_raw)
        p = subprocess.Popen(cmd)
        p.communicate()
        nodes *= 2
        seconds_total_smallest_no_of_nodes /= 2.0
        seconds_total_smallest_no_of_nodes += float(seconds_total_smallest_no_of_nodes) * 0.3
        seconds_total_smallest_no_of_nodes = int(seconds_total_smallest_no_of_nodes)
