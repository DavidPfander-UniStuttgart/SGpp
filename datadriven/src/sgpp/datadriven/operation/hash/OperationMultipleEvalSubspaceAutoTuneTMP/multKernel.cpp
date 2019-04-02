// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#include <algorithm>
#include <iomanip>
#include <limits>
#include <utility>
#include "OperationMultipleEvalSubspaceAutoTuneTMP.hpp"

namespace sgpp::datadriven::SubspaceAutoTuneTMP {

void uncachedMultTransposeInner(
    bool isModLinear, sgpp::base::DataMatrix &paddedDataset, size_t paddedDatasetSize, size_t dim,
    size_t curDataStart, SubspaceNodeCombined &subspace, double *curSubspaceSurpluses,
    size_t validIndicesCount,
    std::array<size_t, SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS + SUBSPACEAUTOTUNETMP_VEC_PADDING>
        &validIndices,
    std::array<size_t, SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS + SUBSPACEAUTOTUNETMP_VEC_PADDING>
        &nextSubspaceIndex,
    std::array<double, SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS + SUBSPACEAUTOTUNETMP_VEC_PADDING>
        &componentResults,
    std::vector<double> &partialPhiEvalsSchedule,
    std::vector<uint32_t> &partialIndicesFlatSchedule);

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
void OperationMultipleEvalSubspaceAutoTuneTMP::multImpl(sgpp::base::DataVector &source,
                                                        sgpp::base::DataVector &result,
                                                        const size_t start_index_data,
                                                        const size_t end_index_data) {
  size_t tid = omp_get_thread_num();

  if (tid == 0) {
    this->setCoefficients(source);
  }

#pragma omp barrier

  // partial phi computations, useful if it is assumed that the level only
  // partially changes
  std::vector<double> partialPhiEvalsSchedule(
      (this->dim + 1) *
      (SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS + SUBSPACEAUTOTUNETMP_VEC_PADDING));
  for (size_t i = 0; i < (this->dim + 1) * (SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS +
                                            SUBSPACEAUTOTUNETMP_VEC_PADDING);
       i++) {
    partialPhiEvalsSchedule[i] = 1.0;
  }

  // for faster index flattening, last element is for padding
  std::vector<uint32_t> partialIndicesFlatSchedule(
      (this->dim + 1) *
      (SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS + SUBSPACEAUTOTUNETMP_VEC_PADDING));
  for (size_t i = 0; i < (this->dim + 1) * (SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS +
                                            SUBSPACEAUTOTUNETMP_VEC_PADDING);
       i++) {
    partialIndicesFlatSchedule[i] = 0;
  }

  // data points that evaluate on the current subspace?
  std::array<size_t, SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS + SUBSPACEAUTOTUNETMP_VEC_PADDING>
      validIndices;
  size_t validIndicesCount;

  // current results
  std::array<double, SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS + SUBSPACEAUTOTUNETMP_VEC_PADDING>
      componentResults;

  // tracks the index of the subspace the data point evaluates next (required
  // for subspace skipping, else always +1)
  std::array<size_t, SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS + SUBSPACEAUTOTUNETMP_VEC_PADDING>
      nextSubspaceIndex;

  std::vector<double> listSubspace(this->maxGridPointsOnLevel);

  for (size_t i = 0; i < this->maxGridPointsOnLevel; i += 1) {
    listSubspace[i] = std::numeric_limits<double>::quiet_NaN();
  }

  // process the next chunk of data tuples in parallel
  for (size_t curDataStart = start_index_data; curDataStart < end_index_data;
       curDataStart += SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS) {
    for (size_t i = 0;
         i < SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS + SUBSPACEAUTOTUNETMP_VEC_PADDING; i++) {
      nextSubspaceIndex[i] = 0.0;
      componentResults[i] = 0.0;
    }

    for (size_t subspaceIndex = 0; subspaceIndex < subspaceCount; subspaceIndex += 1) {
      SubspaceNodeCombined &subspace = this->allSubspaceNodes[subspaceIndex];
      double *curSubspaceSurpluses = nullptr;

      // prepare the subspace array for a list type subspace
      if (subspace.type == SubspaceNodeCombined::SubspaceType::LIST) {
        // fill with surplusses
        for (std::pair<uint32_t, double> tuple : subspace.indexFlatSurplusPairs) {
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
      size_t paddingSize =
          std::min(static_cast<size_t>(validIndicesCount + SUBSPACEAUTOTUNETMP_VEC_PADDING),
                   static_cast<size_t>(SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS +
                                       SUBSPACEAUTOTUNETMP_VEC_PADDING));

      for (size_t i = validIndicesCount; i < paddingSize; i += 1) {
        size_t threadId = SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS + (i - validIndicesCount);
        validIndices[i] = threadId;
        componentResults[threadId] = 0.0;
        nextSubspaceIndex[threadId] = 0;

        for (size_t j = 0; j < this->dim; j += 1) {
          partialPhiEvalsSchedule[(this->dim + 1) * threadId + j] = 1.0;
          partialIndicesFlatSchedule[(this->dim + 1) * threadId + j] = 0.0;
        }
      }

      uncachedMultTransposeInner(isModLinear, paddedDataset, paddedDatasetSize, dim, curDataStart,
                                 subspace, curSubspaceSurpluses, validIndicesCount, validIndices,
                                 nextSubspaceIndex, componentResults, partialPhiEvalsSchedule,
                                 partialIndicesFlatSchedule);

      // in case of list subspace, reset used subspace indices to NaN (so that
      // next subspaces have it initialized properly)
      if (subspace.type == SubspaceNodeCombined::SubspaceType::LIST) {
        for (std::pair<uint32_t, double> &tuple : subspace.indexFlatSurplusPairs) {
          listSubspace[tuple.first] = std::numeric_limits<double>::quiet_NaN();
        }
      }
    }  // end iterate grid

    for (size_t i = 0; i < SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS; i += 1) {
      size_t dataIndex = curDataStart + i;
      result.set(dataIndex, componentResults[i]);
    }
  }  // end iterate data chunks
}
}
