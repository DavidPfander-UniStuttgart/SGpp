#!/usr/bin/python3

import subprocess
import shlex
import re
import numpy as np
import check_cluster_assignements
import socket
import os
import math
from sklearn.metrics import adjusted_rand_score

def execute(cmd):
    p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    return out, err

def evaluate(cmd, f_log, is_first_overall, is_first_lambda):
    f_log.write("\n--------------- CMD -----------------\n")
    f_log.write(cmd)
    out, err = execute(cmd)
    o = out.decode('ascii')
    e = err.decode('ascii')
    f_log.write("\n---------------- OUTPUT -----------------\n")
    f_log.write(o)
    f_log.write("\n---------------- ERROR ------------------\n")
    f_log.write(e)

    if is_first_lambda:
        generate_b_dur = float(re.search(r"last_duration_generate_b: (.*?)s", o).group(1))
        density_total_dur = float(re.search(r"density_duration_total: (.*?)s", o).group(1))
    else:
        generate_b_dur = 0.0
        density_total_dur = 0.0
    if is_first_overall:
        create_graph_dur = float(re.search(r"last_duration_create_graph: (.*?)s", o).group(1))
    else:
        create_graph_dur = 0.0
    prune_graph_dur = float(re.search(r"last_duration_prune_graph: (.*?)s", o).group(1))
    find_cluster_dur = float(re.search(r"find_cluster_duration_total: (.*?)s", o).group(1))
    # total without I/O, kernels only
    total_dur = generate_b_dur + density_total_dur + create_graph_dur + prune_graph_dur + find_cluster_dur

    detected_clusters=int(re.search(r"detected clusters: (.*?)\n", o).group(1))
    datapoints_clusters=int(re.search(r"datapoints in clusters: (.*?)\n", o).group(1))
    # score=float(re.search(r"score: (.*?)\n", o).group(1))
    # if use_distributed_clustering:
    #     elapsed_time=float(re.search(r"elapsed time: (.*?)s\n", o).group(1))
    # else:
    #     elapsed_time=float(re.search(r"total_duration: (.*?)\n", o).group(1))

    if is_first_lambda:
        iterations=float(re.search(r"Number of iterations: (.*?) \(", o).group(1))
    else:
        iterations=0
    f_log.write("\n--------------- SUMMARY -----------------\n")
    f_log.write("detected_clusters: " + str(detected_clusters) + "\n")
    f_log.write("datapoints_clusters: " + str(datapoints_clusters) + "\n")
    # f_log.write("score: " + str(score) + "\n")
    f_log.write("total_dur: " + str(total_dur) + "\n")
    f_log.write("\n--------------- END RUN -----------------\n")
    f_log.flush()
    return detected_clusters, datapoints_clusters, total_dur, iterations # score,

# files contain cluster assignments
def check_assignment(reference_file, results_file):
    # Check existence
    if not os.path.exists(reference_file):
        print("Error! Reference file", reference_file, " does not exist. Exiting...")
        sys.exit(1)
    if not os.path.exists(results_file):
        print("Error! File", results_file, " does not exist. Exiting...")
        sys.exit(1)

    # Load last column of files
    reference_assignement = np.genfromtxt(reference_file, delimiter=',', usecols=(-1))
    actual_assignement = np.genfromtxt(results_file, delimiter=',', usecols=(-1))
    # Check whether number of data points checks out
    if reference_assignement.shape[0] != actual_assignement.shape[0]:
        print("Error! Number of data points in the reference file does not match the number of",
              "data points in the output file. Exiting...")
        sys.exit(1)
    ARI = adjusted_rand_score(reference_assignement, actual_assignement)

    # Calculate hit rate
    verbose = False
    print_cluster_threshold = 1
    counter_correct = check_cluster_assignements.count_correct_cluster_hits(reference_assignement, actual_assignement,
                                                 verbose, print_cluster_threshold)

    return counter_correct / reference_assignement.shape[0], ARI

def get_single_node_cmd_pattern(is_first_overall, is_first_lambda):
    if is_first_overall:
        return "./datadriven/examplesOCL/clustering_cmd --config {config} --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c{num_clusters}_size{dataset_size}_dim{dim}{noise} --level={level} --lambda={lambda_value} --k {k} --epsilon {epsilon} --threshold {threshold} --print_cluster_sizes 1 --target_clusters {num_clusters} --write_all --file_prefix tune_gaussian_c{num_clusters}_size{dataset_size}_dim{dim}{noise} >> tune_precision_{dim}d_{dataset_size}s.log 2>tune_precision_{dim}d_{dataset_size}s_error.log"
    elif is_first_lambda:
        return "./datadriven/examplesOCL/clustering_cmd --config {config} --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c{num_clusters}_size{dataset_size}_dim{dim}{noise} --level={level} --lambda={lambda_value} --k {k} --epsilon {epsilon} --threshold {threshold} --print_cluster_sizes 1 --target_clusters {num_clusters} --reuse_knn_graph tune_gaussian_c{num_clusters}_size{dataset_size}_dim{dim}{noise}_graph.csv --write_all --file_prefix tune_gaussian_c{num_clusters}_size{dataset_size}_dim{dim}{noise} >> tune_precision_{dim}d_{dataset_size}s.log 2>tune_precision_{dim}d_{dataset_size}s_error.log"
    else:
        return "./datadriven/examplesOCL/clustering_cmd --config {config} --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c{num_clusters}_size{dataset_size}_dim{dim}{noise} --level={level} --lambda={lambda_value} --k {k} --epsilon {epsilon} --threshold {threshold} --print_cluster_sizes 1 --target_clusters {num_clusters} --reuse_knn_graph tune_gaussian_c{num_clusters}_size{dataset_size}_dim{dim}{noise}_graph.csv --reuse_density_grid tune_gaussian_c{num_clusters}_size{dataset_size}_dim{dim}{noise}_density_grid.serialized --write_all --file_prefix tune_gaussian_c{num_clusters}_size{dataset_size}_dim{dim}{noise} >> tune_precision_{dim}d_{dataset_size}s.log 2>tune_precision_{dim}d_{dataset_size}s_error.log"

threshold_intervals = 10.0
threshold_start = 0.0
threshold_stop = 1500.0
thresholds_initial = np.linspace(threshold_start, threshold_stop, threshold_intervals)
# threshold_intervals = 5.0
threshold_step_min = 10.0

lambda_intervals = 3
lambda_initial=1E-7
# lambda_initials = [1E-4, 1E-5, 1E-6, 1E-7, 1E-8]
lambda_iterations = int(3) # how many descents

# target_score=0.98

num_tasks=9

hostname=socket.gethostname()
print("hostname:", hostname)
config_map={"argon-gtx": "OCL_configs/config_ocl_float_gtx1080ti.cfg", "argon-tesla2": "OCL_configs/config_ocl_float_QuadroGP100.cfg", "argon-tesla1": "OCL_configs/config_ocl_float_P100.cfg", "pcsgs09": "OCL_configs/config_ocl_float_i76700k.cfg"}
config = config_map[hostname]
mpi_config="clustering_scripts/argon_job_scripts/GTXConf8.cfg"
num_clusters=100
# dataset_size=int(1E6)
dim=10
noise="_noise"
# level=4
clusters_size_level_map = {(10, int(1E6)): 6, (100, int(1E6)): 6, (10, int(1E7)): 7, (100, int(1E7)): 7}
# lambda_value=1E-6
k=6
epsilon=1E-2

use_distributed_clustering_map = {"argon-gtx": False, "argon-tesla2": False, "argon-tesla1": False, "pcsgs09": False}
use_distributed_clustering = use_distributed_clustering_map[hostname]


for dataset_size in [1E7]: # , 1E7
    dataset_size = int(dataset_size)
    datapoints_clusters_min = int(0.75 * dataset_size)
    is_first_overall = True
    level = clusters_size_level_map[(num_clusters, dataset_size)]
    logfile_name = "tune_clustering_" + str(dataset_size) + "s_" + str(num_clusters) + "c" + str(noise) + "_" + str(level) + "l" + ".log"
    print("logfile_name:", logfile_name)
    f_log = open(logfile_name, "w")
    resultsfile_name = "tune_clustering_results_" + str(dataset_size) + "s_"+ str(num_clusters) + "c" + str(noise) + "_" + str(level) + "l" + ".csv"
    print("resultsfile_name:", resultsfile_name)
    f_results = open(resultsfile_name, "w")
    f_results.write("lambda_value, threshold, percent_correct, ARI, duration, iterations\n")

    overall_best_percent_correct = -1
    overall_best_lambda_value = -1
    overall_best_threshold = -1

    # lambda_values = list(lambda_initials)
    lambda_start_lower = lambda_initial
    lambda_factor = 10.0
    lambda_iteration = 0

    # for lambda_value in [1E-4, 1E-5, 1E-6, 1E-7, 1E-8]:
    while lambda_iteration < lambda_iterations:
        f_log.write("lambda_factor: " + str(lambda_factor) + " lambda_start_lower:" + str(lambda_start_lower) + " overall_best_lambda_value: " + str(overall_best_lambda_value) + "\n")
        f_log.flush()
        lambda_values = list(reversed([lambda_start_lower * lambda_factor**i for i in range(0, lambda_intervals)]))
        print(lambda_values)
        if lambda_iteration > 0:
            # cut of left-most, right-most and middle value, as those have already been investigated
            # print("lambda_values unpruned:", lambda_values)
            middle_index = len(lambda_values) // 2 + 1
            lambda_values = lambda_values[1:-1]
        f_log.write("lambda_values:" + str(lambda_values) + "\n")

        for lambda_value in lambda_values:
            is_first_lambda = True
            ########################## threshold #################################
            thresholds = list(thresholds_initial)
            best_threshold = thresholds[0]
            best_percent_correct = -1.0
            threshold_step = thresholds_initial[1] - thresholds_initial[0]
            # do bisection
            while threshold_step > threshold_step_min:
                f_log.write("thresholds:" + str(thresholds) + ", threshold_step:" + str(threshold_step) + "\n")
                f_log.flush()
                for threshold in thresholds:
                    if threshold == 0.0:
                        continue
                    print("lambda_value: " + str(lambda_value) + " threshold: " + str(threshold))
                    if use_distributed_clustering:
                        cmd_pattern = "mpirun.openmpi -n {num_tasks} ./datadriven/examplesMPI/distributed_clustering_cmd --config {config} --MPIconfig {mpi_config} --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c{num_clusters}_size{dataset_size}_dim{dim}{noise} --level={level} --lambda={lambda_value} --k {k} --epsilon {epsilon} --threshold {threshold} --print_cluster_sizes 1 --target_clusters {num_clusters} --write_cluster_map gaussian_c{num_clusters}_size{dataset_size}_dim{dim}{noise}_cluster_map.csv  >> tune_precision_{dim}d_{dataset_size}s.log 2>tune_precision_{dim}d_{dataset_size}s_error.log"
                    else: # clustering_cmd
                        if is_first_overall:
                            f_log.write("is_first_overall: True\n")
                        else:
                            f_log.write("is_first_overall: False\n")
                        if is_first_lambda:
                            f_log.write("is_first_lambda: True\n")
                        else:
                            f_log.write("is_first_lambda: False\n")
                        cmd_pattern = get_single_node_cmd_pattern(is_first_overall, is_first_lambda)
                        # cmd_pattern = "./datadriven/examplesOCL/clustering_cmd --config {config} --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c{num_clusters}_size{dataset_size}_dim{dim}{noise} --level={level} --lambda={lambda_value} --k {k} --epsilon {epsilon} --threshold {threshold} --print_cluster_sizes 1 --target_clusters {num_clusters} --write_all --file_prefix tune_gaussian_c{num_clusters}_size{dataset_size}_dim{dim}{noise} >> tune_precision_{dim}d_{dataset_size}s.log 2>tune_precision_{dim}d_{dataset_size}s_error.log"

                    cmd = cmd_pattern.format(num_tasks=num_tasks, config=config, mpi_config=mpi_config, num_clusters=num_clusters, dataset_size=dataset_size, dim=dim, noise=noise, level=level, lambda_value=lambda_value, k=k, epsilon=epsilon, threshold=threshold)
                    detected_clusters, datapoints_clusters, duration, iterations = evaluate(cmd, f_log, is_first_overall, is_first_lambda) # score,
                    if is_first_overall:
                        is_first_overall = False
                        is_first_lambda = False
                    elif is_first_lambda:
                        is_first_lambda = False

                    results_file = "tune_gaussian_c{num_clusters}_size{dataset_size}_dim{dim}{noise}_cluster_map.csv".format(num_clusters=num_clusters, dataset_size=dataset_size, dim=dim, noise=noise)
                    reference_file = "../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c{num_clusters}_size{dataset_size}_dim{dim}{noise}_class.arff".format(num_clusters=num_clusters, dataset_size=dataset_size, dim=dim, noise=noise)
                    percent_correct, ARI = check_assignment(reference_file, results_file)
                    f_results.write(str(lambda_value) + "," + str(threshold) + "," + str(percent_correct) + "," + str(ARI) + "," + str(duration) + "," + str(iterations) + "\n")
                    f_results.flush()
                    f_log.write("attempt lambda: " + str(lambda_value) + " threshold: " + str(threshold) + " percent_correct: " + str(percent_correct) + " ARI: " + str(ARI) + "\n")
                    f_log.flush()

                    if percent_correct > best_percent_correct:
                        best_percent_correct = percent_correct
                        best_threshold = threshold
                        f_log.write("-> new best_percent_correct:" + str(best_percent_correct) + ", new best_threshold:" + str(best_threshold) + "\n")
                        f_log.flush()

                    # early abort if too many data points are pruned
                    if datapoints_clusters < datapoints_clusters_min:
                        f_log.write("datapoints_clusters too low, aborting threshold testing\n")
                        f_log.flush()
                        break

                threshold_start = max(best_threshold - threshold_step, 0.0)
                threshold_stop = best_threshold + threshold_step
                # remove first and last -> already evaluated
                thresholds = np.linspace(threshold_start, threshold_stop , threshold_intervals)[1:-1]
                threshold_step = thresholds[1] - thresholds[0]
            ########################### end threshold #############################
            if best_percent_correct > overall_best_percent_correct:
                overall_best_percent_correct = best_percent_correct
                overall_best_threshold = best_threshold
                overall_best_lambda_value = lambda_value
                f_log.write("-> new overall_best_percent_correct:" + str(overall_best_percent_correct) + ", overall_best_lambda_value: " + str(overall_best_lambda_value) + ", new best_threshold:" + str(overall_best_threshold) + "\n")
                f_log.flush()


        lambda_iteration += 1
        lambda_start_lower = overall_best_lambda_value / lambda_factor
        lambda_start_upper = overall_best_lambda_value * lambda_factor
        lambda_factor = (lambda_start_upper / lambda_start_lower) **( 1.0/float(lambda_intervals - 1))


    f_log.write("final overall_best_percent_correct: " + str(overall_best_percent_correct) + ", overall_best_threshold: " + str(overall_best_threshold) + ", overall_best_lambda_value: " + str(overall_best_lambda_value) + "\n")
    f_log.close()
    f_results.close()
