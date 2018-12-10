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

void connected_components(std::vector<int64_t> &directed, std::vector<size_t> &map, const int64_t N,
                          const int64_t k, const size_t start_index,
                          std::vector<std::set<int64_t>> &all_clusters) {
  std::deque<size_t> stack;
  stack.push_back(start_index);
  std::set<int64_t> cluster_indices;
  std::set<int64_t> join_clusters;
  cluster_indices.insert(start_index);
  while (stack.size() > 0) {
    size_t cur_index = stack.front();
    stack.pop_front();
    // std::cout << "cur_index: " << cur_index << std::endl;
    for (size_t cur_k = 0; cur_k < static_cast<size_t>(k); cur_k += 1) {
      size_t cur_neighbor = directed[cur_index * k + cur_k];
      // std::cout << "cur_neighbor: " << cur_neighbor << std::endl;
      size_t cur_neighbor_cluster_id = map[cur_neighbor];
      if (cur_neighbor_cluster_id == 0) {
        // cur_neighbor is not part yet part of a cluster
        auto[it, is_new] = cluster_indices.insert(cur_neighbor);
        if (is_new) {
          // cur_neighbor was not yet found via a different edge
          // trigger (breath-first) recursion
          stack.push_back(cur_neighbor);
        }  // else was already recursively processed or is at least scheduled
      } else {
        // cur_neighbor is part of a different cluster
        join_clusters.insert(cur_neighbor_cluster_id);
      }
    }
  }
  // std::cout << "cluster_indices (from: " << start_index << "): ";
  // bool first = true;
  // for (const size_t &i : cluster_indices) {
  //   if (first) {
  //     first = false;
  //   } else {
  //     std::cout << ", ";
  //   }
  //   std::cout << i;
  // }
  // std::cout << std::endl;
  // now know the new partial cluster and whether to join it and with which clusters
  // next step: figure out cluster_id
  size_t cluster_id = global_cluster_id;
  // if (join_clusters.size() > 1) {
  // std::cout << "join_clusters.size(): " << join_clusters.size() << std::endl;
  // }
  if (join_clusters.size() == 0) {
    // is a new cluster
    global_cluster_id += 1;
  } else {
    // always use the lowest cluster_id, could avoid this operation by using the first cluster_id
    // cluster_id = std::min_element(join_clusters.begin(), join_clusters.end());
    cluster_id = *(join_clusters.begin());
  }
  // now assign yourself to the cluster_id
  for (size_t node_index : cluster_indices) {
    map[node_index] = cluster_id;
  }
  if (join_clusters.size() > 0) {
    // insert partial cluster into joined cluster
    all_clusters[cluster_id].insert(cluster_indices.begin(), cluster_indices.end());
  } else {
    // std::cout << "cluster_id: " << cluster_id << " all_clusters.size(): " << all_clusters.size()
    //           << std::endl;
    assert(cluster_id == all_clusters.size());
    all_clusters.push_back(std::move(cluster_indices));
  }
  // reassign all other clusters, expect for the one everyone joins with
  for (size_t other_cluster_id : join_clusters) {
    if (cluster_id == other_cluster_id) {
      continue;
    }
    auto &other_cluster_indices = all_clusters[other_cluster_id];
    // reassign cluster_ids
    for (size_t node_index : other_cluster_indices) {
      map[node_index] = cluster_id;
    }
    // add indices to joined cluster
    all_clusters[cluster_id].insert(other_cluster_indices.begin(), other_cluster_indices.end());
    // mark (partial) cluster as dead
    // all_clusters[other_cluster_id].clear(); // clear does not reduce capacity!

    all_clusters[other_cluster_id] = std::set<int64_t>();
  }
}

}  // namespace detail

void connected_components(std::vector<int64_t> &directed, const int64_t k, std::vector<size_t> &map,
                          std::vector<std::set<int64_t>> &all_clusters) {
  const size_t N = directed.size() / k;
  detail::global_cluster_id = 1;
  // std::vector<size_t> map(N, 0);
  map.resize(N);
  std::fill(map.begin(), map.end(), 0);
  all_clusters.clear();
  // std::vector<std::set<int64_t>> all_clusters;
  all_clusters.emplace_back();  // add empty first dummy set for reserved cluster_id
  for (size_t i = 0; i < N; i += 1) {
    if (map[i] == 0) {
      detail::connected_components(directed, map, N, k, i, all_clusters);
    }
  }
  size_t num_clusters = 0;
  for (size_t i = 0; i < all_clusters.size(); i += 1) {
    if (all_clusters[i].size() > 0) {
      num_clusters += 1;
    }
  }
  std::cout << "num temp clusters (one is dummy!): " << all_clusters.size() << std::endl;
  std::vector<std::set<int64_t>> all_clusters_trimmed;
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
