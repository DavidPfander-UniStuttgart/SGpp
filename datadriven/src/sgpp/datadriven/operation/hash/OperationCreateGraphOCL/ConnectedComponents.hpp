#pragma once

#include <cinttypes>
#include <cstddef>
#include <set>
#include <vector>

namespace sgpp {
namespace datadriven {
namespace clustering {

void connected_components(std::vector<int64_t> &directed, const int64_t k, std::vector<size_t> &map,
                          std::vector<std::set<int64_t>> &all_clusters);

void connected_components_no_set(std::vector<int64_t> &directed, const int64_t k,
                                 std::vector<int64_t> &map,
                                 std::vector<std::vector<int64_t>> &all_clusters);

}  // namespace clustering
}  // namespace datadriven
}  // namespace sgpp
