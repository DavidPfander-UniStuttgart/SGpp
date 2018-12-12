#include "ConnectedComponents.hpp"

#include <cassert>
#include <deque>
#include <iostream>
#include <vector>

namespace sgpp {
namespace datadriven {
namespace clustering {

namespace detail {

size_t global_cluster_id;

void connected_components(std::vector<int64_t> &directed, std::vector<int64_t> &map,
                          const int64_t N, const int64_t k, const size_t start_index,
                          std::vector<std::vector<int64_t>> &all_clusters) {
  bool has_neighbor = false; // true if at least one neighbor exists
  for (size_t cur_k = 0; cur_k < static_cast<size_t>(k); cur_k += 1) {
    int64_t cur_neighbor = directed[start_index * k + cur_k];
    if (cur_neighbor >= 0) {
      has_neighbor = true;
    }
  }
  if (!has_neighbor) {
    // so that removed nodes can be identified during cluster matching
    map[start_index] = -2;
    return;
  }

  std::deque<size_t> stack;
  stack.push_back(start_index);
  std::vector<int64_t> cluster_indices;
  std::set<int64_t> join_clusters;
  cluster_indices.push_back(start_index);
  while (stack.size() > 0) {
    size_t cur_index = stack.front();

    stack.pop_front();
    for (size_t cur_k = 0; cur_k < static_cast<size_t>(k); cur_k += 1) {
      int64_t cur_neighbor = directed[cur_index * k + cur_k];
      if (cur_neighbor < 0) {
        continue;
      }
      int64_t cur_neighbor_cluster_id = map[cur_neighbor];
      if (cur_neighbor_cluster_id == 0) {
        cluster_indices.push_back(cur_neighbor);
        map[cur_neighbor] = -10;  // mark as visited, but not assigned, avoids dups
        stack.push_back(cur_neighbor);
      } else {
        // might be of the visited category
        if (cur_neighbor_cluster_id > 0) {
          // cur_neighbor is part of a different cluster
          join_clusters.insert(cur_neighbor_cluster_id);
        }
      }
    }
  }
  // std::cout << "cluster_indices.size(): " << cluster_indices.size() << std::endl;
  // if (cluster_indices.size() == 1) {
  //   std::cout << "start_index: " << start_index << std::endl;
  //   for (int64_t i : cluster_indices) {
  //     std::cout << i << " ";
  //   }
  //   std::cout << std::endl;
  // }

  // now know the new partial cluster and whether to join it and with which clusters
  // next step: figure out cluster_id
  int64_t cluster_id = global_cluster_id;
  if (join_clusters.size() == 0) {
    // is a new cluster
    global_cluster_id += 1;
  } else {
    // always use the lowest cluster_id, could avoid this operation by using the first cluster_id
    // cluster_id = std::min_element(join_clusters.begin(), join_clusters.end());
    cluster_id = *(join_clusters.begin());
  }
  // now assign yourself to the cluster_id
  for (int64_t node_index : cluster_indices) {
    map[node_index] = cluster_id;
  }
  if (join_clusters.size() > 0) {
    // insert partial cluster into joined cluster
    // std::cout << "cluster_id: " << cluster_id << std::endl;
    auto &append_cluster = all_clusters[cluster_id];
    append_cluster.insert(append_cluster.begin(), cluster_indices.begin(), cluster_indices.end());
  } else {
    assert(cluster_id == all_clusters.size());
    all_clusters.push_back(std::move(cluster_indices));
  }
  // reassign all other clusters, expect for the one everyone joins with
  for (int64_t other_cluster_id : join_clusters) {
    if (cluster_id == other_cluster_id) {
      continue;
    }
    auto &other_cluster_indices = all_clusters[other_cluster_id];
    // reassign cluster_ids
    for (int64_t node_index : other_cluster_indices) {
      map[node_index] = cluster_id;
    }
    // add indices to joined cluster
    auto &append_cluster = all_clusters[cluster_id];
    append_cluster.insert(append_cluster.end(), other_cluster_indices.begin(),
                          other_cluster_indices.end());
    // mark (partial) cluster as dead
    // all_clusters[other_cluster_id].clear(); // clear does not reduce capacity!
    all_clusters[other_cluster_id] = std::vector<int64_t>();
  }
}

}  // namespace detail

void connected_components(std::vector<int64_t> &directed, const int64_t k,
                          std::vector<int64_t> &map,
                          std::vector<std::vector<int64_t>> &all_clusters) {
  const int64_t N = directed.size() / k;
  detail::global_cluster_id = 1;
  map.resize(N);
  std::fill(map.begin(), map.end(), 0);
  all_clusters.clear();
  all_clusters.emplace_back();  // add empty first dummy set for reserved cluster_id
  for (int64_t i = 0; i < N; i += 1) {
    if (map[i] == 0) {
      detail::connected_components(directed, map, N, k, i, all_clusters);
    }
  }
  std::vector<std::vector<int64_t>> all_clusters_trimmed;
  for (size_t i = 0; i < all_clusters.size(); i += 1) {
    if (all_clusters[i].size() > 0) {
      all_clusters_trimmed.push_back(std::move(all_clusters[i]));
    }
  }
  std::swap(all_clusters, all_clusters_trimmed);
}

}  // namespace clustering
}  // namespace datadriven
}  // namespace sgpp
