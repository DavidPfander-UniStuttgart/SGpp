#!/usr/bin/python3

import subprocess
import shlex
import re
import numpy as np
import check_cluster_assignements
import socket
import os

def execute(cmd):
    p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    return out, err

def evaluate(cmd, f_log):
    f_log.write("\n--------------- CMD -----------------\n")
    f_log.write(cmd)
    out, err = execute(cmd)
    o = out.decode('ascii')
    e = err.decode('ascii')
    f_log.write("\n---------------- OUTPUT -----------------\n")
    f_log.write(o)
    f_log.write("\n---------------- ERROR ------------------\n")
    f_log.write(e)
    detected_clusters=int(re.search(r"detected clusters: (.*?)\n", o).group(1))
    datapoints_clusters=int(re.search(r"datapoints in clusters: (.*?)\n", o).group(1))
    score=float(re.search(r"score: (.*?)\n", o).group(1))
    if use_distributed_clustering:
        elapsed_time=float(re.search(r"elapsed time: (.*?)s\n", o).group(1))
    else:
        elapsed_time=float(re.search(r"total_duration: (.*?)\n", o).group(1))
    f_log.write("\n--------------- SUMMARY -----------------\n")
    f_log.write("detected_clusters: " + str(detected_clusters) + "\n")
    f_log.write("datapoints_clusters: " + str(datapoints_clusters) + "\n")
    f_log.write("score: " + str(score) + "\n")
    f_log.write("elapsed_time: " + str(elapsed_time) + "\n")
    f_log.write("\n--------------- END RUN -----------------\n")
    f_log.flush()
    return score, detected_clusters, datapoints_clusters

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

    # Calculate hit rate
    verbose = False
    print_cluster_threshold = 1
    counter_correct = check_cluster_assignements.count_correct_cluster_hits(reference_assignement, actual_assignement,
                                                 verbose, print_cluster_threshold)

    return counter_correct / reference_assignement.shape[0]

threshold_intervals = 10.0
threshold_start = 0.0
threshold_stop = 1000.0

thresholds_initial = np.linspace(threshold_start, threshold_stop, threshold_intervals)
# threshold_intervals = 5.0
threshold_step_min = 1.0


print(thresholds_initial)

target_score=0.98

num_tasks=9

hostname=socket.gethostname()
print("hostname:", hostname)
config_map={"argon-gtx": "config_ocl_float_gtx1080ti.cfg", "argon-tesla2": "config_ocl_float_QuadroGP100.cfg", "argon-tesla1": "config_ocl_float_P100.cfg"}
config = config_map[hostname]
mpi_config="argon_job_scripts/GTXConf8.cfg"
num_clusters=100
dataset_size=int(1E6)
dim=10
noise="_noise"
# level=4
level_map = {int(1E6): 7, int(1E7): 8}
# lambda_value=1E-6
k=6
epsilon=1E-3

datapoints_clusters_min = 0.75 * dataset_size
use_distributed_clustering_map = {"argon-gtx": True, "argon-tesla2": False, "argon-tesla1": False}
use_distributed_clustering = use_distributed_clustering_map[hostname]

for dataset_size in [1E6, 1E7]:
    dataset_size = int(dataset_size)
    level = level_map[dataset_size]
    logfile_name = "tune_clustering_" + str(dataset_size) + "s" + str(noise) + ".log"
    print("logfile_name:", logfile_name)
    f_log = open(logfile_name, "w")
    resultsfile_name = "tune_clustering_results_" + str(dataset_size) + "s" + str(noise) + ".csv"
    print("resultsfile_name:", resultsfile_name)
    f_results = open(resultsfile_name, "w")
    f_results.write("lambda_value, threshold, percent_correct\n")



    if use_distributed_clustering:
        cmd_pattern = "mpirun.openmpi -n {num_tasks} ./datadriven/examplesMPI/distributed_clustering_cmd --config {config} --MPIconfig {mpi_config} --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c{num_clusters}_size{dataset_size}_dim{dim}{noise} --level={level} --lambda={lambda_value} --k {k} --epsilon {epsilon} --threshold {threshold} --print_cluster_sizes 1 --target_clusters {num_clusters} --write_cluster_map gaussian_c{num_clusters}_size{dataset_size}_dim{dim}{noise}_cluster_map.csv  >> tune_precision_{dim}d_{dataset_size}s.log 2>tune_precision_{dim}d_{dataset_size}s_error.log"
    else: # clustering_cmd
        cmd_pattern = "./datadriven/examplesOCL/clustering_cmd --config {config} --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c{num_clusters}_size{dataset_size}_dim{dim}{noise} --level={level} --lambda={lambda_value} --k {k} --epsilon {epsilon} --threshold {threshold} --print_cluster_sizes 1 --target_clusters {num_clusters} --write_cluster_map --file_prefix gaussian_c{num_clusters}_size{dataset_size}_dim{dim}{noise} >> tune_precision_{dim}d_{dataset_size}s.log 2>tune_precision_{dim}d_{dataset_size}s_error.log"

    overall_best_score = -1
    overall_best_lambda_value = -1
    overall_best_threshold = -1
    for lambda_value in [1E-5, 1E-6, 1E-7]:
        thresholds = list(thresholds_initial)
        best_threshold = thresholds[0]
        best_score = -1.0
        threshold_step = thresholds_initial[1] - thresholds_initial[0]
        # do bisection
        while threshold_step > threshold_step_min:
            f_log.write("thresholds:" + str(thresholds) + "threshold_step:" + str(threshold_step) + "\n")
            for threshold in thresholds:
                cmd = cmd_pattern.format(num_tasks=num_tasks, config=config, mpi_config=mpi_config, num_clusters=num_clusters, dataset_size=dataset_size, dim=dim, noise=noise, level=level, lambda_value=lambda_value, k=k, epsilon=epsilon, threshold=threshold)
                score, detected_clusters, datapoints_clusters = evaluate(cmd, f_log)
                # early abort if too many data points are pruned
                if datapoints_clusters < datapoints_clusters_min:
                    f_log.write("datapoints_clusters too low\n")
                    break
                results_file = "gaussian_c{num_clusters}_size{dataset_size}_dim{dim}{noise}_cluster_map.csv".format(num_clusters=num_clusters, dataset_size=dataset_size, dim=dim, noise=noise)
                reference_file = "../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c{num_clusters}_size{dataset_size}_dim{dim}{noise}_class.arff".format(num_clusters=num_clusters, dataset_size=dataset_size, dim=dim, noise=noise)
                percent_correct = check_assignment(reference_file, results_file)
                f_results.write(str(lambda_value) + "," + str(threshold) + "," + str(percent_correct) + "\n")
                print("attempt lambda:", lambda_value, "threshold:", threshold, "percent_correct:", percent_correct)
                f_results.flush()

                if percent_correct > best_score:
                    best_score = percent_correct
                    best_threshold = threshold
                    f_log.write("-> new best_score:" + str(best_score) + "new best_threshold:" + str(best_threshold) + "\n")


            threshold_start = max(best_threshold - threshold_step, 0.0)
            threshold_stop = best_threshold + threshold_step
            # remove first and last -> already evaluated
            thresholds = np.linspace(threshold_start, threshold_stop , threshold_intervals)[1:-1]
            threshold_step = thresholds[1] - thresholds[0]
        if best_score > overall_best_score:
            overall_best_score = best_score
            overall_best_threshold = best_threshold
            overall_best_lambda_value = lambda_value

    f_log.write("overall_best_score:" + str(overall_best_score) + "overall_best_threshold:" + str(overall_best_threshold) + "overall_best_lambda_value:" + str(overall_best_lambda_value) + "\n")
    f_log.close()
    f_results.close()
