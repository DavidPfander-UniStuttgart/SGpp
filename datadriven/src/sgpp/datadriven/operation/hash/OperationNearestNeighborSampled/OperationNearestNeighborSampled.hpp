#pragma once

#include "sgpp/base/datatypes/DataMatrix.hpp"
#include "sgpp/globaldef.hpp"

#include <vector>

namespace sgpp {
namespace datadriven {

class OperationNearestNeighborSampled {
private:
  base::DataMatrix dataset;
  size_t dataset_count;
  size_t dim;

  double last_duration_s;

public:
  OperationNearestNeighborSampled(base::DataMatrix dataset, size_t dim);

  std::vector<int32_t> kNN_lsh(uint32_t k, uint64_t lsh_tables,
                                uint64_t lsh_hashes, double lsh_w);

  std::vector<int32_t> kNN_naive(uint32_t k, std::string configFileName);
};

} // namespace datadriven
} // namespace sgpp
