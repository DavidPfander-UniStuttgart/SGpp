#pragma once

#include <cinttypes>
#include <cstddef>
#include <vector>

namespace sgpp {
namespace datadriven {
namespace clustering {

std::vector<size_t> find_clusters_recursive(std::vector<int64_t> &graph, size_t k);

/// Recursive depth-first function for traversing the k nearest neighbor graph
// parameters:
// index index of data point for which to calculate its cluster
// nodes the graph in a datasetsize * k format, k entries refer to the neighbors
// cluster number of clusters -> index of the current cluster!
// k size of the neighborhood
// clusterList size of the cluster belonging to a specific data points???
// overwrite ??? -> maybe to merge clusters?
size_t find_neighbors(size_t index, std::vector<int64_t> &nodes, size_t cluster, size_t k,
                      std::vector<size_t> &clusterList, bool overwrite = false);

}  // namespace clustering
}  // namespace datadriven
}  // namespace sgpp
