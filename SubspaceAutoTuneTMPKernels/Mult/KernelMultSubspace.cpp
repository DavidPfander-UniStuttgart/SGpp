// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#include "../../datadriven/src/sgpp/datadriven/operation/hash/OperationMultipleEvalSubspaceAutoTuneTMP/SubspaceNode.hpp"
// #include "../SubspaceAutoTuneTMPParameters.hpp"
#include "../calculateIndex.hpp"
#include "autotune_kernel.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sgpp/base/datatypes/DataMatrix.hpp>
#include <sgpp/base/datatypes/DataVector.hpp>
#include <utility>

using namespace sgpp::datadriven::SubspaceAutoTuneTMP;

namespace sgpp::datadriven::SubspaceAutoTuneTMP {

void uncachedMultInner(
    bool isModLinear, sgpp::base::DataMatrix &paddedDataset,
    size_t paddedDatasetSize, size_t dim, size_t curDataStart,
    SubspaceNode &subspace, double *curSubspaceSurpluses,
    size_t validIndicesCount,
    std::array<size_t, SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS +
                           SUBSPACEAUTOTUNETMP_VEC_PADDING> &validIndices,
    std::array<size_t, SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS +
                           SUBSPACEAUTOTUNETMP_VEC_PADDING> &nextSubspaceIndex,
    std::array<double, SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS +
                           SUBSPACEAUTOTUNETMP_VEC_PADDING> &componentResults,
    std::vector<double> &partialPhiEvalsSchedule,
    std::vector<uint32_t> &partialIndicesFlatSchedule) {
  // iterate the indices of the datapoints that evaluate on this subspace
  for (size_t validIndex = 0; validIndex < validIndicesCount;
       validIndex += SUBSPACEAUTOTUNETMP_VEC_PADDING) {
    size_t parallelIndices[4];
    parallelIndices[0] = validIndices[validIndex];
    parallelIndices[1] = validIndices[validIndex + 1];
    parallelIndices[2] = validIndices[validIndex + 2];
    parallelIndices[3] = validIndices[validIndex + 3];

#if SUBSPACEAUTOTUNETMP_ENABLE_PARTIAL_RESULT_REUSAGE == 1
    size_t nextIterationToRecalc = subspace.arriveDiff;
#else
    size_t nextIterationToRecalc = 0;
#endif

    // for vector-gather, points to the data points
    const double *const dataTuplePtr[4] = {
        paddedDataset.data() + (curDataStart + parallelIndices[0]) * dim,
        paddedDataset.data() + (curDataStart + parallelIndices[1]) * dim,
        paddedDataset.data() + (curDataStart + parallelIndices[2]) * dim,
        paddedDataset.data() + (curDataStart + parallelIndices[3]) * dim};

    double *evalIndexValues[4];
    evalIndexValues[0] =
        partialPhiEvalsSchedule.data() + (dim + 1) * parallelIndices[0];
    evalIndexValues[1] =
        partialPhiEvalsSchedule.data() + (dim + 1) * parallelIndices[1];
    evalIndexValues[2] =
        partialPhiEvalsSchedule.data() + (dim + 1) * parallelIndices[2];
    evalIndexValues[3] =
        partialPhiEvalsSchedule.data() + (dim + 1) * parallelIndices[3];

    // for faster index flattening, last element is for padding
    uint32_t *partialIndicesFlat[4];
    partialIndicesFlat[0] =
        partialIndicesFlatSchedule.data() + (dim + 1) * parallelIndices[0];
    partialIndicesFlat[1] =
        partialIndicesFlatSchedule.data() + (dim + 1) * parallelIndices[1];
    partialIndicesFlat[2] =
        partialIndicesFlatSchedule.data() + (dim + 1) * parallelIndices[2];
    partialIndicesFlat[3] =
        partialIndicesFlatSchedule.data() + (dim + 1) * parallelIndices[3];

    uint32_t indexFlat[4];
    double phiEval[4];

#if SUBSPACEAUTOTUNETMP_UNROLL == 1
    size_t parallelIndices2[4];
    parallelIndices2[0] = validIndices[validIndex + 4];
    parallelIndices2[1] = validIndices[validIndex + 5];
    parallelIndices2[2] = validIndices[validIndex + 6];
    parallelIndices2[3] = validIndices[validIndex + 7];

    const double *const dataTuplePtr2[4] = {
        paddedDataset.data() + (curDataStart + parallelIndices2[0]) * dim,
        paddedDataset.data() + (curDataStart + parallelIndices2[1]) * dim,
        paddedDataset.data() + (curDataStart + parallelIndices2[2]) * dim,
        paddedDataset.data() + (curDataStart + parallelIndices2[3]) * dim};

    double *evalIndexValues2[4];
    evalIndexValues2[0] =
        partialPhiEvalsSchedule.data() + (dim + 1) * parallelIndices2[0];
    evalIndexValues2[1] =
        partialPhiEvalsSchedule.data() + (dim + 1) * parallelIndices2[1];
    evalIndexValues2[2] =
        partialPhiEvalsSchedule.data() + (dim + 1) * parallelIndices2[2];
    evalIndexValues2[3] =
        partialPhiEvalsSchedule.data() + (dim + 1) * parallelIndices2[3];

    uint32_t *partialIndicesFlat2[4];
    partialIndicesFlat2[0] =
        partialIndicesFlatSchedule.data() + (dim + 1) * parallelIndices2[0];
    partialIndicesFlat2[1] =
        partialIndicesFlatSchedule.data() + (dim + 1) * parallelIndices2[1];
    partialIndicesFlat2[2] =
        partialIndicesFlatSchedule.data() + (dim + 1) * parallelIndices2[2];
    partialIndicesFlat2[3] =
        partialIndicesFlatSchedule.data() + (dim + 1) * parallelIndices2[3];

    uint32_t indexFlat2[4];
    double phiEval2[4];

    calculateIndex2(isModLinear, dim, nextIterationToRecalc, dataTuplePtr,
                    dataTuplePtr2, subspace.hInverse, partialIndicesFlat,
                    partialIndicesFlat2, evalIndexValues, evalIndexValues2,
                    indexFlat, indexFlat2, phiEval, phiEval2);
#else
    calculateIndex(isModLinear, dim, nextIterationToRecalc, dataTuplePtr,
                   subspace.hInverse, partialIndicesFlat, evalIndexValues,
                   indexFlat, phiEval);
#endif

    double surplus[4];
    surplus[0] = curSubspaceSurpluses[indexFlat[0]];
    surplus[1] = curSubspaceSurpluses[indexFlat[1]];
    surplus[2] = curSubspaceSurpluses[indexFlat[2]];
    surplus[3] = curSubspaceSurpluses[indexFlat[3]];

#if SUBSPACEAUTOTUNETMP_UNROLL == 1
    double surplus2[4];
    surplus2[0] = curSubspaceSurpluses[indexFlat2[0]];
    surplus2[1] = curSubspaceSurpluses[indexFlat2[1]];
    surplus2[2] = curSubspaceSurpluses[indexFlat2[2]];
    surplus2[3] = curSubspaceSurpluses[indexFlat2[3]];
#endif

    for (size_t innerIndex = 0; innerIndex < 4; innerIndex++) {
      if (!std::isnan(surplus[innerIndex])) {
        componentResults[validIndices[validIndex + innerIndex]] +=
            phiEval[innerIndex] * surplus[innerIndex];
        nextSubspaceIndex[validIndices[validIndex + innerIndex]] += 1;
      } else {
#if SUBSPACEAUTOTUNETMP_ENABLE_SUBSPACE_SKIPPING == 1
        // skip to next relevant subspace
        nextSubspaceIndex[validIndices[validIndex + innerIndex]] =
            subspace.jumpTargetIndex;
#else
        nextSubspaceIndex[validIndices[validIndex + innerIndex]] += 1;
#endif
      }
    }

#if SUBSPACEAUTOTUNETMP_UNROLL == 1
    for (size_t innerIndex = 0; innerIndex < 4; innerIndex++) {
      if (!std::isnan(surplus2[innerIndex])) {
        componentResults[validIndices[validIndex + innerIndex + 4]] +=
            phiEval2[innerIndex] * surplus2[innerIndex];
        nextSubspaceIndex[validIndices[validIndex + innerIndex + 4]] += 1;
      } else {
#if SUBSPACEAUTOTUNETMP_ENABLE_SUBSPACE_SKIPPING == 1
        // skip to next relevant subspace
        nextSubspaceIndex[validIndices[validIndex + innerIndex + 4]] =
            subspace.jumpTargetIndex;
#else
        nextSubspaceIndex[validIndices[validIndex + innerIndex + 4]] += 1;
#endif
      }
    }

#endif
  } // end SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS
}

} // namespace sgpp::datadriven::SubspaceAutoTuneTMP

/**
 * Internal mult operator, should not be called directly.
 *
 * @see OperationMultipleEval
 *
 * @param source source operand for the operator
 * @param result stores the result
 * @param start_index_data beginning of the range to process
 * @param end_index_data end of the range to process
 */
AUTOTUNE_EXPORT void KernelMultSubspace(
    size_t maxGridPointsOnLevel, bool isModLinear,
    sgpp::base::DataMatrix &paddedDataset, size_t paddedDatasetSize,
    std::vector<SubspaceNode> &allSubspaceNodes, sgpp::base::DataVector &source,
    sgpp::base::DataVector &result, size_t start_index_data,
    size_t end_index_data) {
  //   size_t tid = omp_get_thread_num();
  //   if (tid == 0) {
  //     setCoefficients(source);
  //   }
  // #pragma omp barrier
  size_t dim = paddedDataset.getNcols();

  // partial phi computations, useful if it is assumed that the level only
  // partially changes
  std::vector<double> partialPhiEvalsSchedule(
      (dim + 1) * (SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS +
                   SUBSPACEAUTOTUNETMP_VEC_PADDING));
  for (size_t i = 0; i < (dim + 1) * (SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS +
                                      SUBSPACEAUTOTUNETMP_VEC_PADDING);
       i++) {
    partialPhiEvalsSchedule[i] = 1.0;
  }

  // for faster index flattening, last element is for padding
  std::vector<uint32_t> partialIndicesFlatSchedule(
      (dim + 1) * (SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS +
                   SUBSPACEAUTOTUNETMP_VEC_PADDING));
  for (size_t i = 0; i < (dim + 1) * (SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS +
                                      SUBSPACEAUTOTUNETMP_VEC_PADDING);
       i++) {
    partialIndicesFlatSchedule[i] = 0;
  }

  // data points that evaluate on the current subspace?
  std::array<size_t, SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS +
                         SUBSPACEAUTOTUNETMP_VEC_PADDING>
      validIndices;
  size_t validIndicesCount;

  // current results
  std::array<double, SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS +
                         SUBSPACEAUTOTUNETMP_VEC_PADDING>
      componentResults;

  // tracks the index of the subspace the data point evaluates next (required
  // for subspace skipping, else always +1)
  std::array<size_t, SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS +
                         SUBSPACEAUTOTUNETMP_VEC_PADDING>
      nextSubspaceIndex;

  std::vector<double> listSubspace(maxGridPointsOnLevel);

  for (size_t i = 0; i < maxGridPointsOnLevel; i += 1) {
    listSubspace[i] = std::numeric_limits<double>::quiet_NaN();
  }

  // process the next chunk of data tuples in parallel
  for (size_t curDataStart = start_index_data; curDataStart < end_index_data;
       curDataStart += SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS) {
    for (size_t i = 0; i < SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS +
                               SUBSPACEAUTOTUNETMP_VEC_PADDING;
         i++) {
      nextSubspaceIndex[i] = 0.0;
      componentResults[i] = 0.0;
    }

    for (size_t subspaceIndex = 0; subspaceIndex < allSubspaceNodes.size() - 1;
         subspaceIndex += 1) {
      SubspaceNode &subspace = allSubspaceNodes[subspaceIndex];
      double *curSubspaceSurpluses = nullptr;

      // prepare the subspace array for a list type subspace
      if (subspace.type == SubspaceNode::SubspaceType::LIST) {
        // fill with surplusses
        for (std::pair<uint32_t, double> tuple :
             subspace.indexFlatSurplusPairs) {
          // actual values are utilized, but only read
          listSubspace[tuple.first] = tuple.second;
        }

        curSubspaceSurpluses = listSubspace.data();
      } else {
        curSubspaceSurpluses = subspace.subspaceArray.data();
      }

      validIndicesCount = 0;

      // figure out which components evaluate on the current subspace
      for (size_t i = 0; i < SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS; i += 1) {
        if (nextSubspaceIndex[i] == subspaceIndex) {
          validIndices[validIndicesCount] = i;
          validIndicesCount += 1;
        }
      }

      // ensure that at least |vector|-many elements evaluate
      // TODO: looks like a potentially too large padding
      size_t paddingSize = std::min(
          static_cast<size_t>(validIndicesCount +
                              SUBSPACEAUTOTUNETMP_VEC_PADDING),
          static_cast<size_t>(SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS +
                              SUBSPACEAUTOTUNETMP_VEC_PADDING));

      for (size_t i = validIndicesCount; i < paddingSize; i += 1) {
        size_t threadId =
            SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS + (i - validIndicesCount);
        validIndices[i] = threadId;
        componentResults[threadId] = 0.0;
        nextSubspaceIndex[threadId] = 0;

        for (size_t j = 0; j < dim; j += 1) {
          partialPhiEvalsSchedule[(dim + 1) * threadId + j] = 1.0;
          partialIndicesFlatSchedule[(dim + 1) * threadId + j] = 0.0;
        }
      }

      uncachedMultInner(isModLinear, paddedDataset, paddedDatasetSize, dim,
                        curDataStart, subspace, curSubspaceSurpluses,
                        validIndicesCount, validIndices, nextSubspaceIndex,
                        componentResults, partialPhiEvalsSchedule,
                        partialIndicesFlatSchedule);

      // for (size_t i = 0; i < SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS; i +=
      // 1) {
      //   if (i > 0) {
      //     std::cout << ", ";
      //   }
      //   std::cout << componentResults[i];
      //   if (std::isnan(componentResults[i])) {
      //     std::terminate();
      //   }
      //   if (componentResults[i] != componentResults[i]) {
      //     std::terminate();
      //   }
      // }
      // std::cout << std::endl;
      // for (size_t i = 0; i < SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS; i +=
      // 1) {
      //   if (std::isnan(componentResults[i])) {
      //     std::terminate();
      //   }
      // }

      // in case of list subspace, reset used subspace indices to NaN (so that
      // next subspaces have it initialized properly)
      if (subspace.type == SubspaceNode::SubspaceType::LIST) {
        for (std::pair<uint32_t, double> &tuple :
             subspace.indexFlatSurplusPairs) {
          listSubspace[tuple.first] = std::numeric_limits<double>::quiet_NaN();
        }
      }
    } // end iterate grid

    for (size_t i = 0; i < SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS; i += 1) {
      size_t dataIndex = curDataStart + i;
      result.set(dataIndex, componentResults[i]);
    }
  } // end iterate data chunks
}
