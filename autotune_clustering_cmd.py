from check_cluster_assignements import count_correct_cluster_hits
import argparse
import os
import subprocess
import atexit
import numpy as np
import json
import time
import filecmp
import sys

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
    parser.add_argument('--verbose', type=str, required=True,
                        help=' [boolean] Print verbose analysis instead of just the overall hitrate)')
    parser.add_argument('--print_cluster_threshold', type=int, required=False, default=20,
                        help=' [int] Do not print clusters containing less data points than this \
                        threshold. Instead they will be summarized as one entry. ')
    parser.add_argument('--file_prefix', type=str, required=True,
                        help=' [filename] File prefix for the clustering_cmd output files.')
    parser.add_argument('--initial_knn_file', type=str, required=False, default="",
                        help=' [filename] File that contains the KNN for the current dataset.\
    If non is given it will be calculated in the first iteration!')
    parser.add_argument('--initial_density_grid_file', type=str, required=False, default="",
                        help=' [filename] File that contains the grid informations.\
    If non is given it will be calculated when needed!')
    parser.add_argument('--initial_density_coef_file', type=str, required=False, default="",
                        help=' [filename] File that contains the density coefficients.\
    If non is given the coefficients will be calculated when needed!')
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
    prefix_arg = "--file_prefix=" + str(args.file_prefix)
    k_arg = "--k=" + str(args.k)
    reuse_graph_arg = ""
    reuse_grid_arg = ""
    reuse_coef_arg = ""
    if args.initial_knn_file != "":
        reuse_graph_arg = "--reuse_knn_graph=" + args.initial_knn_file
    if args.initial_density_grid_file != "":
        reuse_grid_arg = "--reuse_density_grid=" + args.initial_density_grid_file
        if len(args.lambda_list) is not 1:
            print("Error! More than one lambda specified. A range of lambdas only work without\
 an initial density/grid file!")
            print("Length of lambda list: ", len(args.lambda_list))
            sys.exit(1)
    if args.initial_density_coef_file != "":
        reuse_coef_arg = "--reuse_density_coef=" + args.initial_density_coef_file
        if len(args.lambda_list) is not 1:
            print("Error! More than one lambda specified. A range of lambdas only work without\
 an initial density/grid file!")
            print("Length of lambda list: ", len(args.lambda_list))
            sys.exit(1)

    results = []
    crashed = []
    for lambda_value in args.lambda_list:
        # dont reuse coefficients with new lambda
        if args.initial_density_grid_file == "":
            reuse_grid_arg = ""
        if args.initial_density_coef_file == "":
            reuse_coef_arg = ""
        lambda_arg = "--lambda=" + str(lambda_value)
        for threshold_value in args.threshold_list:
            threshold_arg = "--threshold=" + str(threshold_value)
            if args.verbose:
                print("==================================================================")
            print("Running new scenario with lambda: ", lambda_value, " threshold: ", threshold_value, " ...")
            ret = subprocess.call(["datadriven/examplesOCL/clustering_cmd", dataset_arg, k_arg,
                                   config_arg, level_arg, "--epsilon=0.001", threshold_arg, lambda_arg,
                                   "--write_all", reuse_graph_arg, reuse_grid_arg, reuse_coef_arg,
                                   "--knn_algorithm=naive_ocl", prefix_arg],
                                  stdout=subprocess.PIPE)
            if args.verbose is True:
                print("datadriven/examplesOCL/clustering_cmd", dataset_arg, k_arg,
                                    config_arg, level_arg, "--epsilon=0.001", threshold_arg, lambda_arg,
                                    "--write_all", reuse_graph_arg, reuse_grid_arg, reuse_coef_arg,
                                    "--knn_algorithm=naive_ocl", prefix_arg, sep=' ')
            if ret is 0:
                results_file = args.file_prefix + str("_cluster_map.csv")
                if not os.path.exists(results_file):
                    print("Error! File", results_file, " does not exist. Exiting...")
                    sys.exit(1)
                actual_assignement = np.genfromtxt(results_file, delimiter=',', usecols=(-1))
                counter_correct = count_correct_cluster_hits(reference_assignement, actual_assignement,
                                                            args.verbose, args.print_cluster_threshold)
                # Human readable form
                percentage = round(counter_correct/ reference_assignement.shape[0] * 100.0, 4)
                results.append((percentage, (lambda_value, threshold_value)))
                if args.verbose is True:
                    print("------------------------------------------------------------------")
                print("Finished! Correct assignements: ", counter_correct,
                      " => Overall hitrate is: ", percentage, "%")
                if args.initial_knn_file == "":
                    reuse_graph_arg = "--reuse_knn_graph=" + args.file_prefix + "_graph.csv"

                if args.initial_density_grid_file == "":
                    reuse_grid_arg = "--reuse_density_grid=" + args.file_prefix + "_density_grid.serialized"
                if args.initial_density_coef_file == "":
                    reuse_coef_arg = "--reuse_density_coef=" + args.file_prefix + "_density_coef.serialized"
            else:
                print("==> Crashed with this configuration! <==")
                crashed.append((lambda_value, threshold_value))

    sorted_results = sorted(results, key = lambda x : x[0], reverse = True)
    print("==================================================================")
    print("==================================================================")
    print("Results for different parameters:")
    for result in sorted_results:
        print("-> Hitrate ", result[0], " % with lambda ", result[1][0], " and threshold ", result[1][1])
    if len(crashed) > 0:
        print("==================================================================")
        print("Crashed scenarios:")
        for result in crashed:
            print("-> Crashed with lambda ", result[0], " and threshold ", result[1])
