#!/usr/bin/python3

import subprocess, shlex

# master node is added automatically
nodes_start = 128
nodes_end = 128
config_file = "config_ocl_float_P100.cfg"
# config_file = "config_ocl_float_XeonE31585v5.cfg"
# dataset_sizes = [10000000, 100000000]
dataset_sizes = [100000000]
# dataset_sizes = [10000, 100000]
dim = 10
# dataset_file = "datasets/gaussian_c3_size" + str(dataset_size) + "_dim" + str(dim) + ".arff"
level = 8
lambda_value = 1E-6
threshold = 0.7
epsilon = 1E-3

for dataset_size in dataset_sizes:
    dataset_file = "/scratch/snx1600/pfandedd/datasets/gaussian_c3_size" + str(dataset_size) + "_dim" + str(dim)
    nodes = nodes_start
    # for nodes in range(nodes_start, nodes_end):
    while nodes <= nodes_end:
        gpu_constraint = "-C gpumodedefault"
        cmd_raw = "sbatch -N " + str(nodes + 1) + " " + gpu_constraint + " --time 00:25:00 --output=clustering_results/piz_daint_" + str(level) + "l_" + str(dataset_size) +"s_" + str(nodes) + "N.log --error=clustering_results/piz_daint_" + str(level) + "l_" + str(dataset_size) +"s_" + str(nodes) + "N.log_err --wrap=\"srun ./datadriven/examplesMPI/distributed_clustering_cmd --config=" + config_file + " --threshold=" + str(threshold) + " --datasetFileName=" + dataset_file + " --epsilon " + str(epsilon) + "  --level=" + str(level) + " --lambda=" + str(lambda_value) + " --verbose_timers=true\""
        print(cmd_raw)
        cmd = shlex.split(cmd_raw)
        p = subprocess.Popen(cmd)
        p.communicate()
        nodes *= 2
