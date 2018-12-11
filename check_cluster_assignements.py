import argparse
import os
import numpy as np
import sys

def count_correct_cluster_hits(correct_output, actual_output, print_detailed_hitrate, print_ID_assignement):
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
    found_sizes[-2] = 0 # if even we have no removed datapoints we want the bin to exist to avoid ifs
    for i in range(0, actual_output.shape[0]):
        found_sizes[actual_output[i]] =  found_sizes[actual_output[i]] + 1
    print("------------------------------------------------------------------")
    print("Datamining algorithm found ", found_cluster_ids.size - 1, " differenct cluster IDs:")
    for ID in found_cluster_ids:
        # ignore removed datapoints
        if ID != -2:
            print("Size of detected cluster with ID ", ID, ": ", found_sizes[ID])
    print("Number of removed datapoints: ", found_sizes[-2])
    # dict for possible assignements between the two
    bins = {}
    bins[(-1, -2)] = 0 # map noise to removed datapoints even if we do not have noise
    # create bins for each possible assignement
    for key in [(x,y) for x in cluster_ids for y in found_cluster_ids]:
        bins[key] = 0
    # Iterate over data and record hits for each bin
    for i in range(0, actual_output.shape[0]):
        # ignore removed datapoints
        bins[(correct_output[i], actual_output[i])] = bins[(correct_output[i], actual_output[i])] + 1
    #print(bins)

    # Now find the optimal assignement for each cluster id from the
    # dataset. We want to maximize the correct hits (important if clusters
    # of different size were merged by the clustering program)
    correct_assignements = {}
    # stores hits per cluster
    hits = {}
    for key in cluster_ids:
        hits[key] = 0

    correct_assignements[-1] = -2 # hard code noise assignement
    # removed noise points count as a correct hit:
    hits[-1] = bins[(-1, -2)]
    counter_correct = counter_correct + bins[(-1, -2)]
    # find optimal assignement by looking for the bins with the most hits
    for cluster in cluster_ids:
        if cluster == -1: # -1 will be handles later -> hard code assignement with -2
            continue
        current_maximum = -1
        to_remove_key = (0, 0)
        # find current best bin
        for (x,y) in bins:
            if bins[(x,y)] > current_maximum and (int)(y) != -2 and (int)(x) != -1: # ignore mappings onto removed datapoint "cluster"
                current_maximum = bins[(x,y)]
                combination_key = (x,y)
                correct_assignements[x] = y
                to_remove_key = (x, y)
        # Add this bin to correct hits
        hits[to_remove_key[0]] = current_maximum
        counter_correct = counter_correct + current_maximum
        # Remove all assignement combinations made impossible by this assignement
        to_remove_list = []
        for (x,y) in bins:
            if x == to_remove_key[0] or y == to_remove_key[1]:
                to_remove_list.append((x,y)) #cannot del in bins whilst iterating
        for d in to_remove_list:
            del bins[d] # but we can delete in this loop

    if print_ID_mapping:
        print("------------------------------------------------------------------")
        print("Mapping from reference cluster ID to detected cluster ID:")
        print(correct_assignements)

    if print_detailed_hitrate:
        # Get size of clusters from reference file to output percentage of hits
        sizes = {}
        for key in cluster_ids:
            sizes[key] = 0
        for i in range(0, actual_output.shape[0]):
            sizes[correct_output[i]] =  sizes[correct_output[i]] + 1
        print("------------------------------------------------------------------")
        print("Hits per reference cluster ID (", cluster_ids.size - 1, " clusters):")
        for key in cluster_ids:
            if key != -1: #handle noise extra
                percentage = round(hits[key] / sizes[key] * 100.0, 4)
                print("Reference cluster ", key, " hitrate is ", hits[key], " hits out of ", sizes[key], " datapoints => ",
                    percentage, "%")
        percentage = 100.0
        if sizes[-1] > 0: #noise can be zero (the other reference cluster cannot)
            percentage = round(hits[-1] / sizes[-1] * 100.0, 4)
        print("Number of removed noise datapoints is ", hits[-1], " out of ", sizes[-1], " noise datapoints  => ",
            percentage, "%")
    return counter_correct

if __name__ == '__main__':
    # dimensions, clusters, setsize, abweichung, rauschensize
    parser = argparse.ArgumentParser(description='Compares the cluster assignement of each\
    datapoint in a given file with the correct assignement given by a reference file. It will\
    output the overall correct hits and the hits per cluster.')
    parser.add_argument('--reference_cluster_assignement', type=str, required=True,
                        help='File containing the reference clustering assignement (assumed to\
                        be the correct result).')
    parser.add_argument('--cluster_assignement', type=str, required=True,
                        help='File containing the clustering assignement output of the\
                        datamining application.')
    parser.add_argument('--print_hitrate_per_cluster', type=str, required=False, default=True,
                        help='Prints the hitrate per cluster (including the noise)')
    parser.add_argument('--print_ID_mapping', type=str, required=False, default=True,
                        help='Prints how the cluster IDs of the reference file are mapped onto \
                        the cluster IDs of the other file. For example, cluster 1 could be \
                        recognized as cluster 3 in the datamining algorithm, thus we would \
                        print the mapping 1 : 3')
    args = parser.parse_args()
    reference_file = args.reference_cluster_assignement
    results_file = args.cluster_assignement
    print_detailed_hitrate = args.print_hitrate_per_cluster
    print_ID_mapping = args.print_ID_mapping

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

    # Calculate hit rate
    counter_correct = count_correct_cluster_hits(reference_assignement, actual_assignement, print_detailed_hitrate, print_ID_mapping)
    # Human readable form
    percentage = round(counter_correct/ reference_assignement.shape[0] * 100.0, 4)
    print("------------------------------------------------------------------")
    print("Overall hitrate is ", counter_correct, " hits out of ",
    reference_assignement.shape[0], " datapoints  => ", percentage, "%")
