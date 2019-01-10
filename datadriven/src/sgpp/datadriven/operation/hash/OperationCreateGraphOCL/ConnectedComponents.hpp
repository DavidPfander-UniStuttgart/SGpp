#pragma once

#include <cinttypes>
#include <cstddef>
#include <set>
#include <vector>

namespace sgpp {
namespace datadriven {
namespace clustering {

void connected_components(std::vector<int64_t> &directed, const int64_t k,
                          std::vector<int64_t> &map,
                          std::vector<std::vector<int64_t>> &all_clusters,
                          const size_t k_extension_factor = 1);

}  // namespace clustering
}  // namespace datadriven
}  // namespace sgpp
