#include <vector>

namespace sgpp {
namespace datadriven {
namespace clustering {

// directed format: index corresponds to node, k entries per index corresponding to connected
// nodes (edges), -1 indicates empty
std::vector<int64_t> k_inverse_extension(std::vector<int64_t> &directed, const size_t k,
                                         const size_t neighbor_factor = 2);
}  // namespace clustering
}  // namespace datadriven
}  // namespace sgpp
