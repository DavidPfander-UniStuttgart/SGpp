#pragma once

#include <cinttypes>
#include <cstddef>
#include <vector>

namespace sgpp {
namespace datadriven {
namespace clustering {

using neighborhood_list_t = std::vector<std::vector<int64_t>>;

// directed format: index corresponds to node, k entries per index corresponding to connected
// nodes (edges), -1 indicates empty
neighborhood_list_t make_undirected_graph(std::vector<int64_t> &directed, size_t k);

void get_clusters_from_undirected_graph(const neighborhood_list_t &undirected,
                                        std::vector<int64_t> &node_cluster_map,
                                        std::vector<std::vector<int64_t>> &clusters,
                                        size_t cluster_size_min = 0);

void find_clusters(std::vector<int64_t> &directed_graph, size_t k,
                   std::vector<int64_t> &node_cluster_map, neighborhood_list_t &clusters);

}  // namespace clustering
}  // namespace datadriven
}  // namespace sgpp
