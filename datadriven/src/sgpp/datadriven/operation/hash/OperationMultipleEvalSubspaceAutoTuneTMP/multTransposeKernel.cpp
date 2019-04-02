// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#include "OperationMultipleEvalSubspaceAutoTuneTMP.hpp"

namespace sgpp::datadriven::SubspaceAutoTuneTMP {

void listMultInner(bool isModLinear, sgpp::base::DataMatrix &paddedDataset,
                   size_t paddedDatasetSize, size_t dim, sgpp::base::DataVector &alpha,
                   size_t dataIndexBase, size_t end_index_data, SubspaceNodeCombined &subspace,
                   double *levelArrayContinuous, size_t validIndicesCount, size_t *validIndices,
                   size_t *levelIndices, double *evalIndexValuesAll, uint32_t *intermediatesAll);

/**
 * Internal eval operator, should not be called directly.
 *
 * @see OperationMultipleEval
 *
 * @param alpha surplusses of the grid
 * @param result will contain the evaluation results for the given range.
 * @param start_index_data beginning of the range to evaluate
 * @param end_index_data end of the range to evaluate
 */
void OperationMultipleEvalSubspaceAutoTuneTMP::multTransposeImpl(sgpp::base::DataVector &alpha,
                                                                 sgpp::base::DataVector &result,
                                                                 const size_t start_index_data,
                                                                 const size_t end_index_data) {
  size_t tid = omp_get_thread_num();

  if (tid == 0) {
    this->setCoefficients(result);
  }

#pragma omp barrier

  size_t dim = this->paddedDataset.getNcols();
  // const double *const datasetPtr = this->paddedDataset.getPointer();

  size_t totalThreadNumber =
      SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS + SUBSPACEAUTOTUNETMP_VEC_PADDING;

  double *evalIndexValuesAll = new double[(dim + 1) * totalThreadNumber];

  for (size_t i = 0; i < (dim + 1) * totalThreadNumber; i++) {
    evalIndexValuesAll[i] = 1.0;
  }

  // for faster index flattening
  uint32_t *intermediatesAll = new uint32_t[(dim + 1) * totalThreadNumber];

  for (size_t i = 0; i < (dim + 1) * totalThreadNumber; i++) {
    intermediatesAll[i] = 0.0;
  }

  size_t validIndices[SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS + SUBSPACEAUTOTUNETMP_VEC_PADDING];
  size_t validIndicesCount;

  size_t levelIndices[SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS + SUBSPACEAUTOTUNETMP_VEC_PADDING];
  // size_t nextIterationToRecalcReferences[SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS +
  // SUBSPACEAUTOTUNETMP_VEC_PADDING];

  double *listSubspace = new double[this->maxGridPointsOnLevel];

  for (size_t i = 0; i < this->maxGridPointsOnLevel; i++) {
    listSubspace[i] = std::numeric_limits<double>::quiet_NaN();
  }

  /*uint64_t jumpCount = 0;
  uint64_t jumpDistance = 0;
  uint64_t evaluationCounter = 0;
  uint64_t recomputeDimsTotal = 0;
  vector<uint64_t> dimRecalc(dim, 0);*/

  for (size_t dataIndexBase = start_index_data; dataIndexBase < end_index_data;
       dataIndexBase += SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS) {
    for (size_t i = 0; i < totalThreadNumber; i++) {
      levelIndices[i] = 0.0;
      // nextIterationToRecalcReferences[i] = 0;
    }

    for (size_t subspaceIndex = 0; subspaceIndex < subspaceCount; subspaceIndex++) {
      SubspaceNodeCombined &subspace = this->allSubspaceNodes[subspaceIndex];

      // prepare the subspace array for a list type subspace
      if (subspace.type == SubspaceNodeCombined::SubspaceType::LIST) {
        // std::cout << "subspace type is LIST" << std::endl;
        // fill with surplusses
        for (std::pair<uint32_t, double> tuple : subspace.indexFlatSurplusPairs) {
          // accumulator that are later added to the global surplusses
          listSubspace[tuple.first] = 0.0;
        }
      }

      validIndicesCount = 0;

      for (size_t parallelIndex = 0; parallelIndex < SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS;
           parallelIndex++) {
        size_t parallelLevelIndex = levelIndices[parallelIndex];

        if (parallelLevelIndex == subspaceIndex) {
          validIndices[validIndicesCount] = parallelIndex;
          validIndicesCount += 1;
        }
      }

      // padding for up to vector size, no padding required if all data tuples
      // participate as the number of data points is a multiple of the vector
      // width
      size_t paddingSize =
          std::min((int)(validIndicesCount + SUBSPACEAUTOTUNETMP_VEC_PADDING),
                   SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS + SUBSPACEAUTOTUNETMP_VEC_PADDING);

      for (size_t i = validIndicesCount; i < paddingSize; i++) {
        size_t threadId = SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS + (i - validIndicesCount);
        validIndices[i] = threadId;
        levelIndices[threadId] = 0;
        // nextIterationToRecalcReferences[threadId] = 0;
        double *evalIndexValues = evalIndexValuesAll + (dim + 1) * threadId;

        // for faster index flattening, last element is for padding
        uint32_t *intermediates = intermediatesAll + (dim + 1) * threadId;

        for (size_t j = 0; j < dim; j++) {
          evalIndexValues[j] = 1.0;
          intermediates[j] = 0;
        }
      }

      if (subspace.type == SubspaceNodeCombined::SubspaceType::ARRAY) {
        // lock the current subspace, so that no atomic writes are necessary
        subspace.lockSubspace();

        listMultInner(isModLinear, paddedDataset, paddedDatasetSize, dim, alpha, dataIndexBase,
                      end_index_data, subspace, subspace.subspaceArray.data(), validIndicesCount,
                      validIndices, levelIndices, evalIndexValuesAll, intermediatesAll);

        // unlocks the subspace lock for ARRAY and BLUEPRINT type subspaces
        subspace.unlockSubspace();

      } else if (subspace.type == SubspaceNodeCombined::SubspaceType::LIST) {
        listMultInner(isModLinear, paddedDataset, paddedDatasetSize, dim, alpha, dataIndexBase,
                      end_index_data, subspace, listSubspace, validIndicesCount, validIndices,
                      levelIndices, evalIndexValuesAll, intermediatesAll);

        // write results into the global surplus array
        if (subspace.type == SubspaceNodeCombined::SubspaceType::LIST) {
          for (std::pair<uint32_t, double> &tuple : subspace.indexFlatSurplusPairs) {
            if (listSubspace[tuple.first] != 0.0) {
#pragma omp atomic
              tuple.second += listSubspace[tuple.first];
            }

            listSubspace[tuple.first] = std::numeric_limits<double>::quiet_NaN();
          }
        }
      }
    }  // end iterate subspaces
  }    // end iterate chunks

  delete[] evalIndexValuesAll;
  delete[] intermediatesAll;
  delete[] listSubspace;

#pragma omp barrier

  if (tid == 0) {
    this->unflatten(result);
  }
}
}
