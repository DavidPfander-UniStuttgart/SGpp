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
    threshold over a given range for a specific dataset')
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
    for lambda_value in args.lambda_list:
        lambda_arg = "--lambda=" + str(lambda_value)
        for threshold_value in args.threshold_list:
            threshold_arg = "--threshold=" + str(threshold_value)
            print("Running with lambda: ", lambda_value, " threshold: ", threshold_value)
            subprocess.run(["datadriven/examplesOCL/clustering_cmd", dataset_arg, "--k=5",
                            config_arg, level_arg, "--epsilon=0.001", threshold_arg, lambda_arg,
                            "--write_all",
                            "--knn_algorithm=naive_ocl", "--scenario_prefix=test"],
                            stdout=subprocess.PIPE)
            print("Run finished")
            results_file = "test_cluster_map.csv"
            if not os.path.exists(results_file):
                print("Error! File", results_file, " does not exist. Exiting...")
                sys.exit(1)
            print("Counting hits...")
            actual_assignement = np.genfromtxt(results_file, delimiter=',', usecols=(-1))
            counter_correct = count_correct_cluster_hits(reference_assignement, actual_assignement,
                                                        False, False,
                                                         False)
            # Human readable form
            percentage = round(counter_correct/ reference_assignement.shape[0] * 100.0, 4)
            print("Finished! Overall hitrate is: ", percentage)
