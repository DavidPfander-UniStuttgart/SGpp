#pragma once

#include <sgpp/base/datatypes/DataMatrix.hpp>
#include <sgpp/base/datatypes/DataVector.hpp>
#include "SubspaceNode.hpp"

namespace sgpp::datadriven::SubspaceAutoTuneTMP {

void listMultInner(bool isModLinear, sgpp::base::DataMatrix &paddedDataset,
                   size_t paddedDatasetSize, size_t dim, sgpp::base::DataVector &alpha,
                   size_t dataIndexBase, size_t end_index_data, SubspaceNode &subspace,
                   double *levelArrayContinuous, size_t validIndicesCount, size_t *validIndices,
                   size_t *levelIndices, double *evalIndexValuesAll, uint32_t *intermediatesAll);

void multImpl(size_t maxGridPointsOnLevel, bool isModLinear, sgpp::base::DataMatrix &paddedDataset,
              size_t paddedDatasetSize, std::vector<SubspaceNode> &allSubspaceNodes,
              sgpp::base::DataVector &source, sgpp::base::DataVector &result,
              const size_t start_index_data, const size_t end_index_data);

}  // namespace sgpp::datadriven::SubspaceAutoTuneTMP
