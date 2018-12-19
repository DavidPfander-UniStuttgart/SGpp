import argparse
import os
import numpy as np
import sys

def count_correct_cluster_hits(correct_output, actual_output, verbose,
                               print_cluster_threshold):
    # Now count the correct hits .. to do so we must first assigne the
    # clustering IDs from the dataset to the IDs our clustering program produced

    # coutner for correct hits
    counter_correct = 0

    # IDs of clusters from the dataset
    cluster_ids = np.unique(np.append(correct_output, [-1]))
    # IDs of clusters from the clustering program
    found_cluster_ids = np.unique(np.append(actual_output, [-2]))
    # Print found clusters and their sizes
    found_sizes = {}
    for ID in found_cluster_ids:
        found_sizes[ID] = 0
    found_sizes[-2] = 0 # if even we have no removed data points we want the bin to exist to avoid ifs
    for i in range(0, actual_output.shape[0]):
        found_sizes[actual_output[i]] =  found_sizes[actual_output[i]] + 1
    if verbose:
        print("------------------------------------------------------------------")
        print("Datamining algorithm found ", found_cluster_ids.size - 1, " differenct cluster IDs:")
    sum_skipped_clusters = 0
    sum_skipped_datapoints = 0
    for ID in found_cluster_ids:
        # ignore removed data points
        if ID != -2:
            if found_sizes[ID] >= print_cluster_threshold:
                if verbose:
                    print("Size of detected cluster with ID ", (int)(ID), ": ", found_sizes[ID])
            else:
                sum_skipped_clusters = sum_skipped_clusters + 1
                sum_skipped_datapoints = sum_skipped_datapoints + found_sizes[ID]
    print("Number of removed data points: ", found_sizes[-2])
    if sum_skipped_datapoints > 0:
        print("(Ommited output of ", sum_skipped_clusters, " small clusters containing a ",
              "total of ", sum_skipped_datapoints, " data points. A cluster needs to have at least ",
              print_cluster_threshold, " to be printed (--print_cluster_threshold).)")
    # dict for possible assignements between the two
    bins = {}
    bins[(-1, -2)] = 0 # map noise to removed data points even if we do not have noise
    # create bins for each possible assignement
    for key in [(x,y) for x in cluster_ids for y in found_cluster_ids]:
        bins[key] = 0
    # Iterate over data and record hits for each bin
    for i in range(0, actual_output.shape[0]):
        # ignore removed data points
        bins[(correct_output[i], actual_output[i])] = bins[(correct_output[i], actual_output[i])] + 1
    #print(bins)

    # Now find the optimal assignement for each cluster id from the
    # dataset. We want to maximize the correct hits (important if clusters
    # of different size were merged by the clustering program)
    correct_assignements = {}
    # stores hits per cluster
    hits = {}
    for ID in cluster_ids:
        hits[ID] = 0

    correct_assignements[-1] = -2 # hard code noise assignement
    # removed noise points count as a correct hit:
    hits[-1] = bins[(-1, -2)]
    counter_correct = counter_correct + bins[(-1, -2)]
    # find optimal assignement by looking for the bins with the most hits
    for cluster in cluster_ids:
        if cluster == -1: # -1 will be handles later -> hard code assignement with -2
            continue
        current_maximum = -2
        to_remove_ID = (0, 0)
        # find current best bin
        found_bin = False
        for (x,y) in bins:
            if (int)(y) == -2 or (int)(x) == -1:
                continue
            if bins[(x,y)] > current_maximum : # ignore mappings onto removed data point "cluster"
                current_maximum = bins[(x,y)]
                combination_ID = (x,y)
                correct_assignements[x] = y
                to_remove_ID = (x, y)
                found_bin = True
        if found_bin:
            # Add this bin to correct hits
            hits[to_remove_ID[0]] = current_maximum
            counter_correct = counter_correct + current_maximum
            # Remove all assignement combinations made impossible by this assignement
            to_remove_list = []
            for (x,y) in bins:
                if x == to_remove_ID[0] or y == to_remove_ID[1]:
                    to_remove_list.append((x,y)) #cannot del in bins whilst iterating
            for d in to_remove_list:
                del bins[d] # but we can delete in this loop

    if verbose:
        print("------------------------------------------------------------------")
        print("Mapping from reference cluster ID to detected cluster ID:")
        print(correct_assignements)

    if verbose:
        # Get size of clusters from reference file to output percentage of hits
        sizes = {}
        for ID in cluster_ids:
            sizes[ID] = 0
        for i in range(0, correct_output.shape[0]):
            sizes[correct_output[i]] =  sizes[correct_output[i]] + 1
        print("------------------------------------------------------------------")
        print("Hits per reference cluster ID (", cluster_ids.size - 1, " clusters):")
        for ID in cluster_ids:
            if ID != -1: #handle noise extra
                if sizes[ID] <= 0:
                    print("Size of reference cluster ", ID, " is not greater than 0. This should not happen! Exiting ...")
                    sys.exit(1)
                percentage = round(hits[ID] / sizes[ID] * 100.0, 4)
                print("Reference cluster ", (int)(ID), " hit rate is ", hits[ID], " hits out of ", sizes[ID],
                      " data points => ", percentage, "%")
        percentage = 100.0
        if sizes[-1] > 0: #noise can be zero
            percentage = round(hits[-1] / sizes[-1] * 100.0, 4)
        print("Number of removed noise data points is ", hits[-1], " out of ", sizes[-1], " noise data points  => ",
            percentage, "%")
    return counter_correct

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compares the cluster assignement of each\
    data point in a given file with the correct assignement given by a reference file. It will\
    output the overall correct hits and the hits per cluster.')
    parser.add_argument('--reference_cluster_assignement', type=str, required=True,
                        help='[filename] File containing the reference clustering assignement (assumed to\
                        be the correct result).')
    parser.add_argument('--cluster_assignement', type=str, required=True,
                        help='[filename] File containing the clustering assignement output of the\
                        datamining application.')
    parser.add_argument('--print_cluster_threshold', type=int, required=False, default=20,
                        help=' [int] Do not print clusters containing less data points than this \
                        threshold. Instead they will be summarized as one entry. ')
    parser.add_argument('--verbose', type=bool, required=False, default=True,
                        help=' [boolean] Print verbose analysis instead of just the overall hitrate)')
    args = parser.parse_args()
    reference_file = args.reference_cluster_assignement
    results_file = args.cluster_assignement
    verbose = args.verbose

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
    counter_correct = count_correct_cluster_hits(reference_assignement, actual_assignement,
                                                 verbose, args.print_cluster_threshold)
    # Human readable form
    percentage = round(counter_correct/ reference_assignement.shape[0] * 100.0, 4)
    print("------------------------------------------------------------------")
    print("Overall hit rate is ", counter_correct, " hits out of ",
    reference_assignement.shape[0], " data points  => ", percentage, "%")
