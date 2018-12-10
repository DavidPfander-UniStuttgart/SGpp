#include "DetectComponentsRecursive.hpp"

namespace sgpp {
namespace datadriven {
namespace clustering {

/// Assign a clusterindex for each datapoint using the connected components of the graph
// graph has k * #datapoints entries
std::vector<size_t> find_clusters_recursive(std::vector<int64_t> &graph, size_t k) {
  std::vector<size_t> clusters(graph.size() / k);
  size_t cluster_count = 0;
  std::fill(clusters.begin(), clusters.end(), 0);

  // check the cluster to which each data point belongs, i iterates data points
  for (size_t i = 0; i < clusters.size(); i++) {
    // check whether already assigned/whether data point is has any neighbors
    if (clusters[i] == 0 && graph[i * k] != -1) {
      // assume this is a new cluster
      cluster_count += 1;
      if (clustering::find_neighbors(i, graph, cluster_count, k, clusters) != cluster_count) {
        cluster_count -= 1;
      }
    }
  }
  return clusters;
}

/// Recursive depth-first function for traversing the k nearest neighbor graph
// parameters:
// index index of data point for which to calculate its cluster
// nodes the graph in a datasetsize * k format, k entries refer to the neighbors
// cluster number of clusters -> index of the current cluster!
// k size of the neighborhood
// clusterList size of the cluster belonging to a specific data points???
// overwrite ??? -> maybe to merge clusters?
size_t find_neighbors(size_t index, std::vector<int64_t> &nodes, size_t cluster, size_t k,
                      std::vector<size_t> &clusterList, bool overwrite) {
  clusterList[index] = cluster;  // assign current node to new cluster
  bool overwrite_enabled = overwrite;
  bool removed = true;
  // iterates the neighbors of the current node, k entries
  for (size_t i = index * k; i < (index + 1) * k; i++) {
    if (nodes[i] == -2) continue;  // endes with -2 have been pruned!
    removed = false;
    size_t currIndex = nodes[i];       // current neighbor
    if (nodes[currIndex * k] != -1) {  // does it exist?
      if (clusterList[currIndex] == 0 &&
          !overwrite_enabled) {            // has this node already been assigned?
        clusterList[currIndex] = cluster;  // assign neighbor to current cluster
                                           // recursively look at neighbors
        size_t ret_cluster = find_neighbors(currIndex, nodes, cluster, k, clusterList);
        if (ret_cluster != cluster) { // recursively found existing cluster assignment
          clusterList[index] = cluster;
          overwrite_enabled = true;
          i = index * k;
          continue;
        }
      } else if (!overwrite_enabled && clusterList[currIndex] != cluster) {
        // encountered a node that was already processed and belongs to a cluster
        cluster = clusterList[currIndex];
        clusterList[index] = cluster;
        // children already have the correct cluster assignment
        overwrite_enabled = true;
        i = index * k - 1;
        continue;
      } else {
        // encountered another node that belongs to another cluster (connecting multiple prio
        // separate clusters)
        if (clusterList[currIndex] != cluster) {
          find_neighbors(currIndex, nodes, cluster, k, clusterList, true);
          clusterList[currIndex] = cluster;
        }
      }
    }
  }
  if (removed) {
    clusterList[index] = 0;
    cluster = 0;
  }
  return cluster;
}

}  // namespace clustering
}  // namespace datadriven
}  // namespace sgpp
