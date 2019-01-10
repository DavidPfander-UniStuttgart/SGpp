#include "DetectComponents.hpp"
#include <cassert>
#include <iostream>
#include <numa.h>
#include <omp.h>

namespace sgpp {
namespace datadriven {
namespace clustering {

// directed format: index corresponds to node, k entries per index corresponding
// to connected nodes (edges), -1 indicates empty
neighborhood_list_t make_undirected_graph(std::vector<int64_t> &directed,
                                          size_t k) {
  size_t node_count = directed.size() / k;
  // neighborhood_list_t undirected(node_count, std::vector<int>());
  neighborhood_list_t undirected(node_count);
  // add directed edges
  // int64_t dir_edges = 0;

  // use multiple thread to better saturate the memory channels
  int num_threads = omp_get_max_threads();
  if (numa_available() != -1) {
    int num_numa_nodes = numa_num_task_nodes();
    int num_cpus = numa_num_task_cpus();
    // non-HT on first cpu
    num_threads = (num_cpus / num_numa_nodes) / 2;
    // std::cout << "numa nodes: " << num_numa_nodes << " num_cpus: " << num_cpus
    //           << std::endl;
  }
  // else {
  //   std::cout << "numa not available" << std::endl;
  // }
  int64_t segment_size = node_count / num_threads;
// #pragma omp parallel for num_threads(num_threads)
#pragma omp parallel num_threads(num_threads)
  {

    int tid = omp_get_thread_num();

    // segmentation for first-touch policy to be effective
    int64_t segment_start = segment_size * tid;
    int64_t segment_end;
    if (tid < num_threads - 1) {
      // not last thread
      segment_end = segment_size * (tid + 1);
    } else {
      // remainder assigned to last thread
      segment_end = node_count;
    }

    // for (size_t i = 0; i < node_count; i += 1) {
    for (size_t i = segment_start; i < segment_end; i += 1) {
      undirected[i].reserve(2 * k);
      for (size_t cur_k = 0; cur_k < k; cur_k += 1) {
        if (directed[i * k + cur_k] == -1) {
          continue;
        }
        if (directed[i * k + cur_k] == -2) {
          continue;
        }
        undirected[i].push_back(directed[i * k + cur_k]);
        // dir_edges += 1;
      }
      // std::cout << std::endl;
    }

    // int64_t undir_edges = 0;

    // add reversed edge direction (if not already there)
    // int num_threads = omp_get_num_threads();

    // all threads iterate this
    for (size_t i = 0; i < node_count; i += 1) {
      // not self-modifying: graph is non-reflexive
      // enumerate partners of i
      for (size_t cur_k = 0; cur_k < k; cur_k += 1) {
        int64_t pointee = directed[i * k + cur_k];
        if (pointee < 0) {
          continue;
        }
        if (pointee < segment_start || pointee >= segment_end) {
          continue;
        }
        bool found = false;
        for (size_t cur_k_n = 0; cur_k_n < k; cur_k_n += 1) {
          if (directed[pointee * k + cur_k_n] == static_cast<int64_t>(i)) {
            found = true;
            break;
          }
        }
        if (!found) {
          // not in neighbor list -> add
          undirected[pointee].push_back(i);
          // undir_edges += 1;
        }
      }
    }
  }
  // std::cout << "dir_edges: " << dir_edges << " undir_edges: " << undir_edges
  //           << std::endl;
  return undirected;
}

void get_clusters_from_undirected_graph(
    const neighborhood_list_t &undirected,
    std::vector<int64_t> &node_cluster_map,
    std::vector<std::vector<int64_t>> &clusters, size_t cluster_size_min) {
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
                   std::vector<int64_t> &node_cluster_map,
                   neighborhood_list_t &clusters) {
  // std::vector<int64_t> node_cluster_map;
  // std::vector<std::vector<int64_t>> clusters;

  neighborhood_list_t undirected_graph =
      make_undirected_graph(directed_graph, k);

  get_clusters_from_undirected_graph(undirected_graph, node_cluster_map,
                                     clusters, 2);
}
} // namespace clustering
} // namespace datadriven
} // namespace sgpp
