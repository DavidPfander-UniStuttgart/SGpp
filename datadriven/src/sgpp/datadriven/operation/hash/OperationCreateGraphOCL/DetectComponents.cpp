#include "DetectComponents.hpp"
#include <cassert>

namespace sgpp {
namespace datadriven {
namespace clustering {

// directed format: index corresponds to node, k entries per index corresponding to connected
// nodes (edges), -1 indicates empty
neighborhood_list_t make_undirected_graph(std::vector<int64_t> &directed, size_t k) {
  size_t node_count = directed.size() / k;
  // neighborhood_list_t undirected(node_count, std::vector<int>());
  neighborhood_list_t undirected(node_count);
// add directed edges
// #pragma omp parallel for
  for (size_t i = 0; i < node_count; i += 1) {
    for (size_t cur_k = 0; cur_k < k; cur_k += 1) {
      if (directed[i * k + cur_k] == -1) {
        continue;
      }
      if (directed[i * k + cur_k] == -2) {
        continue;
      }
      undirected[i].push_back(directed[i * k + cur_k]);
    }
    // std::cout << std::endl;
  }

  // add reversed edge direction (if not already there)
  for (size_t i = 0; i < node_count; i += 1) {
    // not self-modifying: graph is non-reflexive
    // enumerate partners of i
    for (int64_t pointee : undirected[i]) {
      bool found = false;
      for (int64_t candidate : undirected[pointee]) {
        if (candidate == static_cast<int64_t>(i)) {
          found = true;
          break;
        }
      }
      if (!found) {
        // not in neighbor list -> add
        undirected[pointee].push_back(i);
      }
    }
  }
  return undirected;
}

void get_clusters_from_undirected_graph(const neighborhood_list_t &undirected,
                                        std::vector<int64_t> &node_cluster_map,
                                        std::vector<std::vector<int64_t>> &clusters,
                                        size_t cluster_size_min) {
  // not yet processd set to -1, solitary set to -2
  node_cluster_map.resize(undirected.size());
  std::fill(node_cluster_map.begin(), node_cluster_map.end(), -1);
  clusters.clear();
  int64_t cluster_id = 0;
  for (size_t i = 0; i < undirected.size(); i++) {
    if (node_cluster_map[i] != -1) {
      // already processed
      continue;
    }
    // not yet processed, part of a new cluster (or solitary)
    std::vector<int64_t> cluster;
    cluster.push_back(i);
    size_t cur_node_index = 0;
    node_cluster_map[i] = cluster_id;
    while (cur_node_index < cluster.size()) {
      for (int64_t neighbor_index : undirected[cluster[cur_node_index]]) {
        if (node_cluster_map[neighbor_index] == -1) {
          // not yet processed, enqueue
          cluster.push_back(neighbor_index);
          node_cluster_map[neighbor_index] = cluster_id;
        } else {
          assert(node_cluster_map[neighbor_index] == cluster_id);
        }
      }
      cur_node_index += 1;
    }

    // now all cluster members are collected
    if (cluster.size() >= cluster_size_min) {
      clusters.push_back(std::move(cluster));
      cluster_id += 1;
    } else {
      // rejected, nodes are solitary
      for (int64_t member_index : cluster) {
        node_cluster_map[member_index] = -2;
      }
    }
  }
}

void find_clusters(std::vector<int64_t> &directed_graph, size_t k,
                   std::vector<int64_t> &node_cluster_map, neighborhood_list_t &clusters) {
  // std::vector<int64_t> node_cluster_map;
  // std::vector<std::vector<int64_t>> clusters;

  neighborhood_list_t undirected_graph = make_undirected_graph(directed_graph, k);

  get_clusters_from_undirected_graph(undirected_graph, node_cluster_map, clusters, 2);
}
}  // namespace clustering
}  // namespace datadriven
}  // namespace sgpp
