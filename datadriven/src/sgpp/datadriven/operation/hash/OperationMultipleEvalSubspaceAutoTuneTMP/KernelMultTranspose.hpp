#pragma once

#include <array>
#include <sgpp/base/datatypes/DataMatrix.hpp>
#include <sgpp/base/datatypes/DataVector.hpp>
#include "SubspaceAutoTuneTMPParameters.hpp"
#include "SubspaceNode.hpp"

namespace sgpp::datadriven::SubspaceAutoTuneTMP {

void uncachedMultTransposeInner(
    bool isModLinear, sgpp::base::DataMatrix &paddedDataset, size_t paddedDatasetSize, size_t dim,
    size_t curDataStart, SubspaceNode &subspace, double *curSubspaceSurpluses,
    size_t validIndicesCount,
    std::array<size_t, SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS + SUBSPACEAUTOTUNETMP_VEC_PADDING>
        &validIndices,
    std::array<size_t, SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS + SUBSPACEAUTOTUNETMP_VEC_PADDING>
        &nextSubspaceIndex,
    std::array<double, SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS + SUBSPACEAUTOTUNETMP_VEC_PADDING>
        &componentResults,
    std::vector<double> &partialPhiEvalsSchedule,
    std::vector<uint32_t> &partialIndicesFlatSchedule);

void multTransposeImpl(size_t maxGridPointsOnLevel, bool isModLinear,
                       sgpp::base::DataMatrix &paddedDataset, size_t paddedDatasetSize,
                       std::vector<SubspaceNode> &allSubspaceNodes, sgpp::base::DataVector &alpha,
                       sgpp::base::DataVector &result, const size_t start_index_data,
                       const size_t end_index_data);

}  // namespace sgpp::datadriven::SubspaceAutoTuneTMP
