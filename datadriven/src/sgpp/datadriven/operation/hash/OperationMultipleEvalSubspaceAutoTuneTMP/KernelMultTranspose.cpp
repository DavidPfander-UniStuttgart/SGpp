// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#include "KernelMultTranspose.hpp"
#include <sgpp/base/datatypes/DataMatrix.hpp>
#include <sgpp/base/datatypes/DataVector.hpp>
#include "SubspaceAutoTuneTMPParameters.hpp"
#include "SubspaceNode.hpp"
#include "calculateIndex.hpp"

namespace sgpp::datadriven::SubspaceAutoTuneTMP {

void listMultTransposeInner(bool isModLinear, sgpp::base::DataMatrix &paddedDataset,
                            size_t paddedDatasetSize, size_t dim, sgpp::base::DataVector &alpha,
                            size_t dataIndexBase, size_t end_index_data, SubspaceNode &subspace,
                            double *levelArrayContinuous, size_t validIndicesCount,
                            size_t *validIndices, size_t *levelIndices, double *evalIndexValuesAll,
                            uint32_t *intermediatesAll) {
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

    const double *const dataTuplePtr[4] = {
        paddedDataset.data() + (dataIndexBase + parallelIndices[0]) * dim,
        paddedDataset.data() + (dataIndexBase + parallelIndices[1]) * dim,
        paddedDataset.data() + (dataIndexBase + parallelIndices[2]) * dim,
        paddedDataset.data() + (dataIndexBase + parallelIndices[3]) * dim};

    double *evalIndexValues[4];
    evalIndexValues[0] = evalIndexValuesAll + (dim + 1) * parallelIndices[0];
    evalIndexValues[1] = evalIndexValuesAll + (dim + 1) * parallelIndices[1];
    evalIndexValues[2] = evalIndexValuesAll + (dim + 1) * parallelIndices[2];
    evalIndexValues[3] = evalIndexValuesAll + (dim + 1) * parallelIndices[3];

    // for faster index flattening, last element is for padding
    uint32_t *intermediates[4];
    intermediates[0] = intermediatesAll + (dim + 1) * parallelIndices[0];
    intermediates[1] = intermediatesAll + (dim + 1) * parallelIndices[1];
    intermediates[2] = intermediatesAll + (dim + 1) * parallelIndices[2];
    intermediates[3] = intermediatesAll + (dim + 1) * parallelIndices[3];

    uint32_t indexFlat[4];
    double phiEval[4];

#if SUBSPACEAUTOTUNETMP_UNROLL == 1
    size_t parallelIndices2[4];
    parallelIndices2[0] = validIndices[validIndex + 4];
    parallelIndices2[1] = validIndices[validIndex + 5];
    parallelIndices2[2] = validIndices[validIndex + 6];
    parallelIndices2[3] = validIndices[validIndex + 7];

    const double *const dataTuplePtr2[4] = {
        paddedDataset.data() + (dataIndexBase + parallelIndices2[0]) * dim,
        paddedDataset.data() + (dataIndexBase + parallelIndices2[1]) * dim,
        paddedDataset.data() + (dataIndexBase + parallelIndices2[2]) * dim,
        paddedDataset.data() + (dataIndexBase + parallelIndices2[3]) * dim};

    double *evalIndexValues2[4];
    evalIndexValues2[0] = evalIndexValuesAll + (dim + 1) * parallelIndices2[0];
    evalIndexValues2[1] = evalIndexValuesAll + (dim + 1) * parallelIndices2[1];
    evalIndexValues2[2] = evalIndexValuesAll + (dim + 1) * parallelIndices2[2];
    evalIndexValues2[3] = evalIndexValuesAll + (dim + 1) * parallelIndices2[3];

    uint32_t *intermediates2[4];
    intermediates2[0] = intermediatesAll + (dim + 1) * parallelIndices2[0];
    intermediates2[1] = intermediatesAll + (dim + 1) * parallelIndices2[1];
    intermediates2[2] = intermediatesAll + (dim + 1) * parallelIndices2[2];
    intermediates2[3] = intermediatesAll + (dim + 1) * parallelIndices2[3];

    uint32_t indexFlat2[4];
    double phiEval2[4];

    calculateIndex2(isModLinear, dim, nextIterationToRecalc, dataTuplePtr, dataTuplePtr2,
                    subspace.hInverse, intermediates, intermediates2, evalIndexValues,
                    evalIndexValues2, indexFlat, indexFlat2, phiEval, phiEval2);
#else
    calculateIndex(isModLinear, dim, nextIterationToRecalc, dataTuplePtr, subspace.hInverse,
                   intermediates, evalIndexValues, indexFlat, phiEval);
#endif

    double surplus[4];
    // for (size_t i = 0; i < 4; i += 1) {
    //   if (indexFlat[i] >= subspace.gridPointsOnLevel) {
    //     std::cout << "l: ";
    //     for (size_t j = 0; j < dim; j += 1) {
    //       if (j > 0) {
    //         std::cout << ", ";
    //       }
    //       std::cout << subspace.level[j];
    //     }
    //     std::cout << std::endl;
    //     std::cout << "hInverse: ";
    //     for (size_t j = 0; j < dim; j += 1) {
    //       if (j > 0) {
    //         std::cout << ", ";
    //       }
    //       std::cout << subspace.hInverse[j];
    //     }
    //     std::cout << std::endl;
    //     std::cout << "dataTuplePtr: ";
    //     for (size_t j = 0; j < dim; j += 1) {
    //       if (j > 0) {
    //         std::cout << ", ";
    //       }
    //       std::cout << dataTuplePtr[i][j];
    //     }
    //     std::cout << std::endl;
    //     std::cout << "intermediates: "
    //               << intermediates[i][nextIterationToRecalc] << std::endl;
    //     std::cout << "evalIndexValues: "
    //               << evalIndexValues[i][nextIterationToRecalc] << std::endl;
    //     std::cout << "phiEval: " << phiEval[i] << std::endl;
    //     std::cout << "existingGridPointsOnLevel: "
    //               << subspace.existingGridPointsOnLevel << std::endl;
    //     std::cout << "nextIterationToRecalc: " << nextIterationToRecalc
    //               << std::endl;
    //     std::cout << "parallelIndices[" << i << "] = " << parallelIndices[i]
    //               << std::endl;
    //     std::cout << "indexFlat[" << i << "]: " << indexFlat[i] << std::endl;
    //     // throw;
    //   }
    // }
    // for (size_t i = 0; i < 4; i += 1) {
    //   if (indexFlat[i] >= subspace.gridPointsOnLevel) {
    //     throw;
    //   }
    // }

    // std::cout << "indexFlat 0: " << indexFlat[0]
    //           << "indexFlat 1: " << indexFlat[1]
    //           << "indexFlat 2: " << indexFlat[2]
    //           << "indexFlat 3: " << indexFlat[3] << std::endl;
    surplus[0] = levelArrayContinuous[indexFlat[0]];
    surplus[1] = levelArrayContinuous[indexFlat[1]];
    surplus[2] = levelArrayContinuous[indexFlat[2]];
    surplus[3] = levelArrayContinuous[indexFlat[3]];

#if SUBSPACEAUTOTUNETMP_UNROLL == 1
    double surplus2[4];
    surplus2[0] = levelArrayContinuous[indexFlat2[0]];
    surplus2[1] = levelArrayContinuous[indexFlat2[1]];
    surplus2[2] = levelArrayContinuous[indexFlat2[2]];
    surplus2[3] = levelArrayContinuous[indexFlat2[3]];
#endif

    for (size_t innerIndex = 0; innerIndex < 4; innerIndex++) {
      size_t parallelIndex = parallelIndices[innerIndex];

      if (!std::isnan(surplus[innerIndex])) {
        double partialSurplus = 0.0;

        if (dataIndexBase + parallelIndex < end_index_data &&
            parallelIndex < SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS) {
          partialSurplus = phiEval[innerIndex] * alpha[dataIndexBase + parallelIndex];

          size_t localIndexFlat = indexFlat[innerIndex];

          // no atomics required, working on temporary arrays
          //#pragma omp atomic
          levelArrayContinuous[localIndexFlat] += partialSurplus;
        }

        // nextIterationToRecalcReferences[parallelIndex] = subspace.nextDiff;
        levelIndices[parallelIndex] += 1;
      } else {
#if SUBSPACEAUTOTUNETMP_ENABLE_SUBSPACE_SKIPPING == 1
        // skip to next relevant subspace
        // nextIterationToRecalcReferences[parallelIndex] = subspace.jumpDiff;
        levelIndices[parallelIndex] = subspace.jumpTargetIndex;
#else
        // nextIterationToRecalcReferences[parallelIndex] = subspace.nextDiff;
        levelIndices[parallelIndex] += 1;
#endif
      }
    }  // end innerIndex

#if SUBSPACEAUTOTUNETMP_UNROLL == 1

    // for second vector
    for (size_t innerIndex = 0; innerIndex < 4; innerIndex++) {
      size_t parallelIndex = parallelIndices2[innerIndex];

      if (!std::isnan(surplus2[innerIndex])) {
        double partialSurplus = 0.0;

        if (dataIndexBase + parallelIndex < end_index_data &&
            parallelIndex < SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS) {
          partialSurplus = phiEval2[innerIndex] * alpha[dataIndexBase + parallelIndex];

          size_t localIndexFlat = indexFlat2[innerIndex];

          // no atomics required, subspace is locked before processing
          //#pragma omp atomic
          levelArrayContinuous[localIndexFlat] += partialSurplus;
        }

        // nextIterationToRecalcReferences[parallelIndex] = subspace.nextDiff;
        levelIndices[parallelIndex] += 1;
      } else {
#if SUBSPACEAUTOTUNETMP_ENABLE_SUBSPACE_SKIPPING == 1
        // skip to next relevant subspace
        // nextIterationToRecalcReferences[parallelIndex] = subspace.jumpDiff;
        levelIndices[parallelIndex] = subspace.jumpTargetIndex;
#else
        // nextIterationToRecalcReferences[parallelIndex] = subspace.nextDiff;
        levelIndices[parallelIndex] += 1;
#endif
      }
    }  // end innerIndex

#endif
  }  // end parallel
}

void multTransposeImpl(size_t maxGridPointsOnLevel, bool isModLinear,
                       sgpp::base::DataMatrix &paddedDataset, size_t paddedDatasetSize,
                       std::vector<SubspaceNode> &allSubspaceNodes, sgpp::base::DataVector &alpha,
                       sgpp::base::DataVector &result, const size_t start_index_data,
                       const size_t end_index_data) {
  // size_t tid = omp_get_thread_num();
  // if (tid == 0) {
  //   setCoefficients(result);
  // }
  // #pragma omp barrier

  size_t dim = paddedDataset.getNcols();

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

  double *listSubspace = new double[maxGridPointsOnLevel];

  for (size_t i = 0; i < maxGridPointsOnLevel; i++) {
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

    for (size_t subspaceIndex = 0; subspaceIndex < allSubspaceNodes.size() - 1; subspaceIndex++) {
      SubspaceNode &subspace = allSubspaceNodes[subspaceIndex];

      // prepare the subspace array for a list type subspace
      if (subspace.type == SubspaceNode::SubspaceType::LIST) {
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

      if (subspace.type == SubspaceNode::SubspaceType::ARRAY) {
        // lock the current subspace, so that no atomic writes are necessary
        subspace.lockSubspace();

        listMultTransposeInner(isModLinear, paddedDataset, paddedDatasetSize, dim, alpha,
                               dataIndexBase, end_index_data, subspace,
                               subspace.subspaceArray.data(), validIndicesCount, validIndices,
                               levelIndices, evalIndexValuesAll, intermediatesAll);

        // unlocks the subspace lock for ARRAY and BLUEPRINT type subspaces
        subspace.unlockSubspace();

      } else if (subspace.type == SubspaceNode::SubspaceType::LIST) {
        listMultTransposeInner(isModLinear, paddedDataset, paddedDatasetSize, dim, alpha,
                               dataIndexBase, end_index_data, subspace, listSubspace,
                               validIndicesCount, validIndices, levelIndices, evalIndexValuesAll,
                               intermediatesAll);

        // write results into the global surplus array
        if (subspace.type == SubspaceNode::SubspaceType::LIST) {
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

  // #pragma omp barrier
  //   if (tid == 0) {
  //     this->unflatten(result);
  //   }
}
}  // namespace sgpp::datadriven::SubspaceAutoTuneTMP
