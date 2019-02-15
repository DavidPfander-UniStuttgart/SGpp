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

#
# configuration is at the end of the file
#

class threshold_configuration:
    def __init__(self, threshold_intervals, threshold_start, threshold_stop, threshold_step_min):
        self.threshold_intervals = threshold_intervals # number of values to generate per iteration
        self.threshold_start = threshold_start
        self.threshold_stop = threshold_stop
        self.threshold_step_min = threshold_step_min # about search if step size is smaller than this

class lambda_configuration:
    def __init__(self, lambda_intervals_initial, lambda_initial, lambda_iterations):
        self.lambda_intervals_initial = lambda_intervals_initial
        self.lambda_intervals = 5 # number of values to generate per iteration
        self.lambda_initial = lambda_initial
        self.lambda_iterations = lambda_iterations # how many descents
        self.lambda_factor = 10.0

class dataset_configuration:
    def __init__(self, num_clusters, dataset_sizes, dim, noise_suffix):
        self.num_clusters = num_clusters
        self.dataset_sizes = dataset_sizes
        self.dim = dim
        self.noise_suffix = noise_suffix

class parameter_tuner:

    def __init__(self):
        # globals
        self.use_distributed_clustering_map = {"argon-gtx": False, "argon-tesla2": False, "argon-tesla1": False, "pcsgs09": False}
        self.num_tasks=9
        self.config_map={"argon-gtx": "OCL_configs/config_ocl_float_gtx1080ti.cfg", "argon-tesla2": "OCL_configs/config_ocl_float_QuadroGP100.cfg", "argon-tesla1": "OCL_configs/config_ocl_float_P100.cfg", "pcsgs09": "OCL_configs/config_ocl_float_i76700k.cfg"}
        self.mpi_config="clustering_scripts/argon_job_scripts/GTXConf8.cfg"
        self.clusters_size_level_map = {(10, int(1E6)): 6, (100, int(1E6)): 7, (10, int(1E7)): 7, (100, int(1E7)): 7}
        self.hostname=socket.gethostname()
        print("hostname:", self.hostname)
        self.config = self.config_map[self.hostname]
        self.use_distributed_clustering = self.use_distributed_clustering_map[self.hostname]

    def execute(self, cmd):
        # print(cmd)
        p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        return out, err

    def evaluate(self, cmd):
        self.f_log.write("\n------------ START RUN --------------\n")
        self.f_log.write("\n--------------- CMD -----------------\n")
        self.f_log.write(cmd)
        out, err = self.execute(cmd)
        o = out.decode('ascii')
        e = err.decode('ascii')
        self.f_log.write("\n-------------- OUTPUT ---------------\n")
        self.f_log.write(o)
        self.f_log.write("\n-------------- ERROR ----------------\n")
        self.f_log.write(e)

        if self.is_first_lambda:
            generate_b_dur = float(re.search(r"last_duration_generate_b: (.*?)s", o).group(1))
            density_total_dur = float(re.search(r"density_duration_total: (.*?)s", o).group(1))
        else:
            generate_b_dur = 0.0
            density_total_dur = 0.0
        if self.is_first_overall:
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

        if self.is_first_lambda:
            iterations=float(re.search(r"Number of iterations: (.*?) \(", o).group(1))
        else:
            iterations=0
        self.f_log.write("\n------------- SUMMARY ---------------\n")
        self.f_log.write("detected_clusters: " + str(detected_clusters) + "\n")
        self.f_log.write("datapoints_clusters: " + str(datapoints_clusters) + "\n")
        self.f_log.write("total_dur: " + str(total_dur) + "\n")
        self.f_log.write("\n------------- END RUN ---------------\n")
        self.f_log.flush()
        return detected_clusters, datapoints_clusters, total_dur, iterations # score,

    # files contain cluster assignments
    def check_assignment(self, reference_file, results_file):
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

    def get_cmd_pattern(self):
        if self.use_distributed_clustering:
            return "mpirun.openmpi -n {num_tasks} ./datadriven/examplesMPI/distributed_clustering_cmd --config {config} --MPIconfig {mpi_config} --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c{num_clusters}_size{dataset_size}_dim{dim}{noise_suffix} --level={level} --lambda={lambda_value} --k {k} --epsilon {epsilon} --threshold {threshold} --print_cluster_sizes 1 --target_clusters {num_clusters} --write_cluster_map gaussian_c{num_clusters}_size{dataset_size}_dim{dim}{noise_suffix}_cluster_map.csv  >> tune_precision_{dim}d_{dataset_size}s.log 2>tune_precision_{dim}d_{dataset_size}s_error.log"
        else: # clustering_cmd
            if self.is_first_overall:
                self.f_log.write("is_first_overall: True\n")
                return "./datadriven/examplesOCL/clustering_cmd --config {config} --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c{num_clusters}_size{dataset_size}_dim{dim}{noise_suffix} --level={level} --lambda={lambda_value} --k {k} --epsilon {epsilon} --threshold {threshold} --target_clusters {num_clusters} --write_all --file_prefix tune_gaussian_c{num_clusters}_size{dataset_size}_dim{dim}{noise_suffix} >> tune_precision_{dim}d_{dataset_size}s.log 2>tune_precision_{dim}d_{dataset_size}s_error.log" # --print_cluster_sizes 1
            else:
                self.f_log.write("is_first_overall: False\n")
            if self.is_first_lambda:
                self.f_log.write("is_first_lambda: True\n")
                return "./datadriven/examplesOCL/clustering_cmd --config {config} --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c{num_clusters}_size{dataset_size}_dim{dim}{noise_suffix} --level={level} --lambda={lambda_value} --k {k} --epsilon {epsilon} --threshold {threshold} --target_clusters {num_clusters} --reuse_knn_graph tune_gaussian_c{num_clusters}_size{dataset_size}_dim{dim}{noise_suffix}_graph.csv --write_all --file_prefix tune_gaussian_c{num_clusters}_size{dataset_size}_dim{dim}{noise_suffix} >> tune_precision_{dim}d_{dataset_size}s.log 2>tune_precision_{dim}d_{dataset_size}s_error.log" # --print_cluster_sizes 1
            else:
                self.f_log.write("is_first_lambda: False\n")
                return "./datadriven/examplesOCL/clustering_cmd --config {config} --binary_header_filename ../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c{num_clusters}_size{dataset_size}_dim{dim}{noise_suffix} --level={level} --lambda={lambda_value} --k {k} --epsilon {epsilon} --threshold {threshold} --target_clusters {num_clusters} --reuse_knn_graph tune_gaussian_c{num_clusters}_size{dataset_size}_dim{dim}{noise_suffix}_graph.csv --reuse_density_grid tune_gaussian_c{num_clusters}_size{dataset_size}_dim{dim}{noise_suffix}_density_grid.serialized --write_all --file_prefix tune_gaussian_c{num_clusters}_size{dataset_size}_dim{dim}{noise_suffix} >> tune_precision_{dim}d_{dataset_size}s.log 2>tune_precision_{dim}d_{dataset_size}s_error.log" #  --print_cluster_sizes 1

    def tune_lambda(self, l_config, t_config, d_config, dataset_size, level, k, epsilon):
        lambda_start_lower = l_config.lambda_initial
        lambda_iteration = 0
        lambda_factor = l_config.lambda_factor # gets updated per iteration

        while lambda_iteration < l_config.lambda_iterations:
            self.f_log.write("lambda_factor: " + str(lambda_factor) + " lambda_start_lower:" + str(lambda_start_lower) + " overall_best_lambda_value: " + str(self.overall_best_lambda_value) + "\n")
            self.f_log.flush()
            if lambda_iteration > 0:
                lambda_values = list(reversed([lambda_start_lower * lambda_factor**i for i in range(0, l_config.lambda_intervals)]))
                print("lambda_values unfiltered:", lambda_values)
                # cut of left-most, right-most and middle value, as those have already been investigated
                # print("lambda_values unpruned:", lambda_values)
                middle_index = len(lambda_values) // 2
                del lambda_values[middle_index]
                lambda_values = lambda_values[1:-1]
            else:
                lambda_values = list(reversed([lambda_start_lower * lambda_factor**i for i in range(0, l_config.lambda_intervals_initial)]))
            self.f_log.write("lambda_values filtered:" + str(lambda_values) + "\n")
            print("lambda_values filtered (or initial):", lambda_values)

            for lambda_value in lambda_values:
                self.is_first_lambda = True
                best_ARI, best_threshold = self.tune_threshold(l_config, t_config, d_config, dataset_size, level, k, epsilon, lambda_value)
                if best_ARI > self.overall_best_ARI:
                    self.overall_best_ARI = best_ARI
                    self.overall_best_threshold = best_threshold
                    self.overall_best_lambda_value = lambda_value
                    self.f_log.write("-> new overall_best_ARI:" + str(self.overall_best_ARI) + ", overall_best_lambda_value: " + str(self.overall_best_lambda_value) + ", new best_threshold:" + str(self.overall_best_threshold) + "\n")
                    self.f_log.flush()
            lambda_iteration += 1
            lambda_start_lower = self.overall_best_lambda_value / lambda_factor
            lambda_start_upper = self.overall_best_lambda_value * lambda_factor
            lambda_factor = (lambda_start_upper / lambda_start_lower) **( 1.0/float(l_config.lambda_intervals - 1))

    def tune_threshold(self, l_config, t_config, d_config, dataset_size, level, k, epsilon, lambda_value):
        thresholds = np.linspace(t_config.threshold_start, t_config.threshold_stop, t_config.threshold_intervals)
        best_threshold = thresholds[0]
        best_ARI = -1.0
        if len(thresholds) > 1:
            threshold_step = thresholds[1] - thresholds[0]
        else:
            # so that one iteration is performed, execution is aborted after single interval
            threshold_step = t_config.threshold_step_min + 1.0

        # do bisection
        while threshold_step > t_config.threshold_step_min:
            print("lambda_value:", lambda_value, "thresholds:", thresholds)
            self.f_log.write("thresholds:" + str(thresholds) + ", threshold_step:" + str(threshold_step) + "\n")
            self.f_log.flush()
            for threshold in thresholds:
                if threshold == 0.0:
                    continue
                print("lambda_value: " + str(lambda_value) + " threshold: " + str(threshold), end=" ")
                cmd_pattern = self.get_cmd_pattern()

                cmd = cmd_pattern.format(num_tasks=self.num_tasks, config=self.config, mpi_config=self.mpi_config, num_clusters=d_config.num_clusters, dataset_size=dataset_size, dim=d_config.dim, noise_suffix=d_config.noise_suffix, level=level, lambda_value=lambda_value, k=k, epsilon=epsilon, threshold=threshold)
                detected_clusters, datapoints_clusters, duration, iterations = self.evaluate(cmd)
                if self.is_first_overall:
                    self.is_first_overall = False
                    self.is_first_lambda = False
                elif self.is_first_lambda:
                    self.is_first_lambda = False

                results_file = "tune_gaussian_c{num_clusters}_size{dataset_size}_dim{dim}{noise_suffix}_cluster_map.csv".format(num_clusters=d_config.num_clusters, dataset_size=dataset_size, dim=d_config.dim, noise_suffix=d_config.noise_suffix)
                reference_file = "../../DissertationCodeTesla1/SGpp/datasets_WPDM18/gaussian_c{num_clusters}_size{dataset_size}_dim{dim}{noise_suffix}_class.arff".format(num_clusters=d_config.num_clusters, dataset_size=dataset_size, dim=d_config.dim, noise_suffix=d_config.noise_suffix)
                percent_correct, ARI = self.check_assignment(reference_file, results_file)
                print("duration:", duration, "ARI:", ARI)
                self.f_results.write(str(lambda_value) + "," + str(threshold) + "," + str(percent_correct) + "," + str(ARI) + "," + str(duration) + "," + str(iterations) + "\n")
                self.f_results.flush()
                self.f_log.write("attempt lambda: " + str(lambda_value) + " threshold: " + str(threshold) + " percent_correct: " + str(percent_correct) + " ARI: " + str(ARI) + "\n")
                self.f_log.flush()

                if ARI > best_ARI:
                    best_ARI = ARI
                    best_threshold = threshold
                    self.f_log.write("-> new best_ARI:" + str(best_ARI) + ", new best_threshold:" + str(best_threshold) + "\n")
                    self.f_log.flush()

                # early abort if too many data points are pruned
                if datapoints_clusters < self.datapoints_clusters_min:
                    self.f_log.write("datapoints_clusters too low, aborting threshold testing\n")
                    self.f_log.flush()
                    break

            # for a single interval only perform a single iteration
            if len(thresholds) == 1:
                return best_ARI, best_threshold
            else:
                threshold_start = max(best_threshold - threshold_step, 0.0)
                threshold_stop = best_threshold + threshold_step
                # remove first and last -> already evaluated
                thresholds = np.linspace(threshold_start, threshold_stop , t_config.threshold_intervals)[1:-1]
                threshold_step = thresholds[1] - thresholds[0]
        return best_ARI, best_threshold

    def tune(self, t_config, l_config, d_config, k, epsilon):
        for dataset_size in d_config.dataset_sizes:
            dataset_size = int(dataset_size)
            self.datapoints_clusters_min = int(0.75 * dataset_size)
            self.is_first_overall = True
            level = self.clusters_size_level_map[(d_config.num_clusters, dataset_size)]
            logfile_name = "tune_clustering_" + str(dataset_size) + "s_" + str(d_config.num_clusters) + "c" + str(d_config.noise_suffix) + "_" + str(level) + "l" + ".log"
            print("logfile_name:", logfile_name)
            self.f_log = open(logfile_name, "w")
            resultsfile_name = "tune_clustering_results_" + str(dataset_size) + "s_"+ str(d_config.num_clusters) + "c" + str(d_config.noise_suffix) + "_" + str(level) + "l" + ".csv"
            print("resultsfile_name:", resultsfile_name)
            self.f_results = open(resultsfile_name, "w")
            self.f_results.write("lambda_value, threshold, percent_correct, ARI, duration, iterations\n")
            self.overall_best_ARI = -1
            self.overall_best_lambda_value = -1
            self.overall_best_threshold = -1
            self.tune_lambda(l_config, t_config, d_config, dataset_size, level, k, epsilon)
            self.f_log.write("final overall_best_ARI: " + str(self.overall_best_ARI) + ", overall_best_threshold: " + str(self.overall_best_threshold) + ", overall_best_lambda_value: " + str(self.overall_best_lambda_value) + "\n")
            self.f_log.close()
            self.f_results.close()

#############################
# configure threshold search
#############################

t_config = threshold_configuration(threshold_intervals = 6.0, threshold_start = 0.0, threshold_stop = 1500.0, threshold_step_min = 50.0)

#############################
# configure lambda search
#############################
l_config = lambda_configuration(lambda_intervals_initial = 3, lambda_initial=1E-7, lambda_iterations = int(3))

#############################
# configure dataset
#############################
d_config = dataset_configuration(num_clusters=10, dataset_sizes=[1E6], dim=10, noise_suffix="_noise")

#############################
# configure kNN and solver
#############################
k=6
epsilon=1E-2

t = parameter_tuner()
t.tune(t_config, l_config, d_config, k, epsilon)
