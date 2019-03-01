#!/usr/bin/python3

import numpy as np
import argparse
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix

parser = argparse.ArgumentParser(description='Compares the cluster assignement of each\
data point in a given file with the correct assignement given by a reference file. It will\
output the overall correct hits and the hits per cluster.')
parser.add_argument('--reference_cluster_assignment', type=str, required=True,
                    help='[filename] File containing the reference clustering assignement (assumed to\
                    be the correct result).')
parser.add_argument('--cluster_assignment', type=str, required=True,
                    help='[filename] File containing the clustering assignement output of the\
                    datamining application.')
args = parser.parse_args()
reference_file = args.reference_cluster_assignment
results_file = args.cluster_assignment

# Load last column of files
reference_assignement = np.genfromtxt(reference_file, delimiter=',', usecols=(-1))
actual_assignement = np.genfromtxt(results_file, delimiter=',', usecols=(-1))

# filter noise indices and remove from both mapping (otherwise ARI tries to optimize noise assignment as well, which is undesirable, as by definition noise is not a cluster)
indices_to_remove = []
for i, x in np.ndenumerate(reference_assignement):
    if x < 0.0:
        indices_to_remove.append(i)
# print(indices_to_remove)
reference_assignement = np.delete(reference_assignement, indices_to_remove)
actual_assignement = np.delete(actual_assignement, indices_to_remove)


# Check whether number of data points checks out
if reference_assignement.shape[0] != actual_assignement.shape[0]:
    print("Error! Number of data points in the reference file does not match the number of",
          "data points in the output file. Exiting...")
    sys.exit(1)
ARI = adjusted_rand_score(reference_assignement, actual_assignement)
print("ARI:", ARI)


M = contingency_matrix(actual_assignement, reference_assignement)
print(M.shape)
ones = np.ones(shape=(M.shape[1], 1))
# print(ones.shape)
sums = np.dot(M, ones)
print(sums.T)

np.set_printoptions(threshold=np.inf)
# print("contingency matrix:")
# print(M)
