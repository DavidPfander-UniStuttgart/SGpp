from check_cluster_assignements import count_correct_cluster_hits
import argparse
import os
import subprocess
import atexit
import numpy as np
import json
import time
import filecmp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autotunes the clustering parameter lambda and\
    threshold over a given range for a specific dataset',
                                     epilog='Example call: python3 autotune_clustering_cmd.py --dataset test.arff --reference_file test-erg.txt --grid_level 4 --config datadriven/examplesOCL/MyOCLConf.cfg --k 8 --lambda_list 0.2 0.1 0.01 0.001 0.0001 --threshold_list 0.0 0.00001 0.0001 0.001 0.01 0.1 1.0 --verbose=True --print_cluster_threshold=0')
    parser.add_argument('--dataset', type=str, required=True,
                        help='[filename] File containing the dataset.')
    parser.add_argument('--reference_file', type=str, required=True,
                        help='[filename] File containing the clustering assignement.')
    parser.add_argument('--grid_level', type=int, required=True,
                        help='[int] Level of the sparse grid')
    parser.add_argument('--config', type=str, required=True,
                        help=' [filename] OpenCL configuration file.')
    parser.add_argument('--k', type=int, required=True,
                        help=' [int] Number of considered neighbors for the KNN algorithm.')
    parser.add_argument('--verbose', type=bool, required=False, default=True,
                        help=' [boolean] Print verbose analysis instead of just the overall hitrate)')
    parser.add_argument('--print_cluster_threshold', type=int, required=False, default=20,
                        help=' [int] Do not print clusters containing less data points than this \
                        threshold. Instead they will be summarized as one entry. ')
    parser.add_argument('--lambda_list', nargs='+', help='Lambda values that will be tested', required=True)
    parser.add_argument('--threshold_list', nargs='+', help='Threshold values that will be tested', required=True)
    args = parser.parse_args()

    # Check existence
    reference_file = args.reference_file
    if not os.path.exists(reference_file):
        print("Error! Reference file", reference_file, " does not exist. Exiting...")
        sys.exit(1)

    # Load last column of files
    reference_assignement = np.genfromtxt(reference_file, delimiter=',', usecols=(-1))

    dataset_arg = "--datasetFileName=" + args.dataset
    config_arg = "--config=" + args.config
    level_arg = "--level=" + str(args.grid_level)
    k_arg = "--k=" + str(args.k)
    reuse_graph_arg = ""
    reuse_grid_arg = ""
    reuse_coef_arg = ""
    results = []
    crashed = []
    for lambda_value in args.lambda_list:
        reuse_grid_arg = ""
        reuse_coef_arg = "" # dont reuse coefficients with new lambda
        lambda_arg = "--lambda=" + str(lambda_value)
        for threshold_value in args.threshold_list:
            threshold_arg = "--threshold=" + str(threshold_value)
            if args.verbose:
                print("==================================================================")
            print("Running new scenario with lambda: ", lambda_value, " threshold: ", threshold_value, " ...")
            ret = subprocess.call(["datadriven/examplesOCL/clustering_cmd", dataset_arg, k_arg,
                                   config_arg, level_arg, "--epsilon=0.001", threshold_arg, lambda_arg,
                                   "--write_all", reuse_graph_arg, reuse_grid_arg, reuse_coef_arg,
                                   "--knn_algorithm=naive_ocl", "--file_prefix=test"],
                                  stdout=subprocess.PIPE)
            print("datadriven/examplesOCL/clustering_cmd", dataset_arg, k_arg,
                                   config_arg, level_arg, "--epsilon=0.001", threshold_arg, lambda_arg,
                                   "--write_all", reuse_graph_arg, reuse_grid_arg, reuse_coef_arg,
                                   "--knn_algorithm=naive_ocl", "--file_prefix=test", sep=' ')
            if ret is 0:
                results_file = "test_cluster_map.csv"
                if not os.path.exists(results_file):
                    print("Error! File", results_file, " does not exist. Exiting...")
                    sys.exit(1)
                actual_assignement = np.genfromtxt(results_file, delimiter=',', usecols=(-1))
                counter_correct = count_correct_cluster_hits(reference_assignement, actual_assignement,
                                                            args.verbose, args.print_cluster_threshold)
                # Human readable form
                percentage = round(counter_correct/ reference_assignement.shape[0] * 100.0, 4)
                results.append((percentage, (lambda_value, threshold_value)))
                if args.verbose:
                    print("------------------------------------------------------------------")
                print("Finished! Correct assignements: ", counter_correct, " => Overall hitrate is: ", percentage)
                reuse_graph_arg = "--reuse_knn_graph=test_graph.csv"
                reuse_grid_arg = "--reuse_density_grid=test_density_grid.serialized"
                reuse_coef_arg = "--reuse_density_coef=test_density_coef.serialized"
            else:
                print("==> Crashed with this configuration! <==")
                crashed.append((lambda_value, threshold_value))

    sorted_results = sorted(results, key = lambda x : x[0], reverse = True)
    print("==================================================================")
    print("==================================================================")
    print("Results for different parameters:")
    for result in sorted_results:
        print("-> Hitrate ", result[0], " %% with lambda ", result[1][0], " and threshold ", result[1][1])
    print("==================================================================")
    print("Crashed scenarios:")
    for result in crashed:
        print("-> Crashed with lambda ", result[0], " and threshold ", result[1])
