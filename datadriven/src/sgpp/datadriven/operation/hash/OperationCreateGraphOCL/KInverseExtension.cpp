#include <cassert>
#include <iostream>

#include "KInverseExtension.hpp"

namespace sgpp {
namespace datadriven {
namespace clustering {

// directed format: index corresponds to node, k entries per index corresponding to connected
// nodes (edges), -1 indicates empty
std::vector<int64_t> k_inverse_extension(std::vector<int64_t> &directed, const size_t k,
                                         const size_t neighbor_factor) {
  const size_t node_count = directed.size() / k;
  const size_t num_neighbors_extended = neighbor_factor * k;
  std::vector<int64_t> extended(node_count * num_neighbors_extended);
  // add directed edges
  for (size_t i = 0; i < node_count; i += 1) {
    for (size_t cur_k = 0; cur_k < k; cur_k += 1) {
      if (directed[i * k + cur_k] < 0) {
        extended[i * num_neighbors_extended + cur_k] = -1;
      } else {
        extended[i * num_neighbors_extended + cur_k] = directed[i * k + cur_k];
      }
    }
    for (size_t cur_k = 0; cur_k < k; cur_k += 1) {
      extended[i * num_neighbors_extended] = -1;
    }
  }
  std::vector<size_t> counters(node_count);
  std::fill(counters.begin(), counters.end(), k);
  size_t edges_left_out = 0;

  // add reversed edge direction (if not already there)
  for (size_t i = 0; i < node_count; i += 1) {
    // not self-modifying: graph is non-reflexive
    // enumerate partners of i
    for (size_t cur_k = 0; cur_k < k; cur_k += 1) {
      int64_t pointee = extended[i * num_neighbors_extended + cur_k];
      if (counters[pointee] < num_neighbors_extended) {
        // make sure to only add new neighbors
        bool found = false;
        for (size_t neighbor_k = 0; neighbor_k < num_neighbors_extended; neighbor_k += 1) {
          if (extended[pointee * num_neighbors_extended + counters[pointee]] == i) {
            found = true;
            break;
          }
        }
        if (!found) {
          extended[pointee * num_neighbors_extended + counters[pointee]] = i;
          counters[pointee] += 1;
        }
      } else {
        edges_left_out += 1;
      }
    }
  }
  std::cout << "edges_left_out: " << edges_left_out << std::endl;
  return extended;
}

}  // namespace clustering
}  // namespace datadriven
}  // namespace sgpp
