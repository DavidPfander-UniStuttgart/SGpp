// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#include "../../datadriven/src/sgpp/datadriven/operation/hash/OperationMultipleEvalSubspaceAutoTuneTMP/SubspaceNode.hpp"
#include <array>
#include <sgpp/base/datatypes/DataMatrix.hpp>
#include <sgpp/base/datatypes/DataVector.hpp>
// #include "../SubspaceAutoTuneTMPParameters.hpp"
#include "../../datadriven/src/sgpp/datadriven/tools/PartitioningTool.hpp"
#include "../calculateIndex.hpp"
#include <cmath>
#include <sgpp/base/grid/GridStorage.hpp>

#include "autotune_kernel.hpp"

using namespace sgpp::datadriven::SubspaceAutoTuneTMP;

namespace sgpp::datadriven::SubspaceAutoTuneTMP::detail {

static inline uint32_t flattenIndex(const size_t dim,
                                    const std::vector<uint32_t> &maxIndices,
                                    const std::vector<uint32_t> &index) {
  uint32_t indexFlat = index[0];
  indexFlat >>= 1;

  for (size_t i = 1; i < dim; i++) {
    uint32_t actualDirectionGridPoints = maxIndices[i];
    actualDirectionGridPoints >>= 1;
    indexFlat *= actualDirectionGridPoints;
    uint32_t actualIndex = index[i];
    actualIndex >>= 1; // divide index by 2, skip even indices
    indexFlat += actualIndex;
  }

  return indexFlat;
}

static inline uint32_t flattenLevel(size_t dim, size_t maxLevel,
                                    std::vector<uint32_t> &level) {
  uint32_t levelFlat = 0;
  levelFlat += level[dim - 1];

  // loop terminates at -1
  for (int i = static_cast<int>(dim - 2); i >= 0; i--) {
    levelFlat *= static_cast<uint32_t>(maxLevel);
    levelFlat += level[i];
  }

  return levelFlat;
}

// writes a result vector in the order of the points in the grid storage
void unflatten(size_t dim, std::vector<SubspaceNode> &allSubspaceNodes,
               std::map<uint32_t, uint32_t> &allLevelsIndexMap,
               sgpp::base::GridStorage &storage, size_t maxLevel,
               sgpp::base::DataVector &result) {
  std::vector<uint32_t> level(dim);
  std::vector<uint32_t> maxIndices(dim);
  std::vector<uint32_t> index(dim);

  base::level_t curLevel;
  base::index_t curIndex;

  for (size_t gridIndex = 0; gridIndex < storage.getSize(); gridIndex++) {
    sgpp::base::GridPoint &point = storage.getPoint(gridIndex);

    for (size_t d = 0; d < dim; d++) {
      point.get(d, curLevel, curIndex);
      level[d] = curLevel;
      index[d] = curIndex;
      maxIndices[d] = 1 << curLevel;
    }

    uint32_t levelFlat = flattenLevel(dim, maxLevel, level);
    uint32_t subspaceIndex = allLevelsIndexMap.find(levelFlat)->second;
    SubspaceNode &subspace = allSubspaceNodes[subspaceIndex];

    uint32_t indexFlat = flattenIndex(dim, maxIndices, index);
    double surplus = subspace.getSurplus(indexFlat);
    if (!std::isnan(surplus)) {
      result.set(gridIndex, surplus);
    }
  }
}

void zeroCoefficients(size_t dim, sgpp::base::GridStorage &storage,
                      std::vector<SubspaceNode> &allSubspaceNodes,
                      std::map<uint32_t, uint32_t> &allLevelsIndexMap,
                      size_t &maxLevel) {
  std::vector<uint32_t> level(dim);
  std::vector<uint32_t> maxIndices(dim);
  std::vector<uint32_t> index(dim);

  base::level_t curLevel;
  base::index_t curIndex;

  for (size_t gridPoint = 0; gridPoint < storage.getSize(); gridPoint += 1) {
    sgpp::base::GridPoint &point = storage.getPoint(gridPoint);

    for (size_t d = 0; d < dim; d += 1) {
      point.get(d, curLevel, curIndex);
      level[d] = curLevel;
      index[d] = curIndex;
      maxIndices[d] = 1 << curLevel;
    }

    uint32_t levelFlat = flattenLevel(dim, maxLevel, level);
    uint32_t indexFlat = flattenIndex(dim, maxIndices, index);
    uint32_t subspaceIndex = allLevelsIndexMap.find(levelFlat)->second;
    SubspaceNode &subspace = allSubspaceNodes[subspaceIndex];
    subspace.setSurplus(indexFlat, 0.0);
  }
}

void listMultTransposeInner(
    bool isModLinear, sgpp::base::DataMatrix &paddedDataset,
    size_t paddedDatasetSize, size_t dim, sgpp::base::DataVector &source,
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

    calculateIndex2(isModLinear, dim, nextIterationToRecalc, dataTuplePtr,
                    dataTuplePtr2, subspace.hInverse, intermediates,
                    intermediates2, evalIndexValues, evalIndexValues2,
                    indexFlat, indexFlat2, phiEval, phiEval2);
#else
    calculateIndex(isModLinear, dim, nextIterationToRecalc, dataTuplePtr,
                   subspace.hInverse, intermediates, evalIndexValues, indexFlat,
                   phiEval);
#endif

    double surplus[4];
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
          partialSurplus =
              phiEval[innerIndex] * source[dataIndexBase + parallelIndex];

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
    } // end innerIndex

#if SUBSPACEAUTOTUNETMP_UNROLL == 1

    // for second vector
    for (size_t innerIndex = 0; innerIndex < 4; innerIndex++) {
      size_t parallelIndex = parallelIndices2[innerIndex];

      if (!std::isnan(surplus2[innerIndex])) {
        double partialSurplus = 0.0;

        if (dataIndexBase + parallelIndex < end_index_data &&
            parallelIndex < SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS) {
          partialSurplus =
              phiEval2[innerIndex] * source[dataIndexBase + parallelIndex];

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
    } // end innerIndex

#endif
  } // end parallel
}

} // namespace sgpp::datadriven::SubspaceAutoTuneTMP::detail

AUTOTUNE_EXPORT sgpp::base::DataVector
KernelMultTransposeSubspace(size_t &maxGridPointsOnLevel, bool &isModLinear,
                            sgpp::base::DataMatrix &paddedDataset,
                            size_t &paddedDatasetSize,
                            sgpp::base::GridStorage &storage,
                            std::vector<SubspaceNode> &allSubspaceNodes,
                            std::map<uint32_t, uint32_t> &allLevelsIndexMap,
                            size_t &maxLevel, sgpp::base::DataVector &source) {

  size_t dim = paddedDataset.getNcols();

  // initialize surpluses to zero in subspace data structure
  sgpp::datadriven::SubspaceAutoTuneTMP::detail::zeroCoefficients(
      dim, storage, allSubspaceNodes, allLevelsIndexMap, maxLevel);

  // std::cout << "source.size(): " << source.size() << std::endl;
  // std::cout << "maxGridPointsOnLevel: " << maxGridPointsOnLevel << std::endl;
  // std::cout << "isModLinear: " << isModLinear << std::endl;
  // std::cout << "maxLevel: " << maxLevel << std::endl;
  // std::cout << "source (first 20): ";
  // for (size_t i = 0; i < 20; i += 1) {
  //   if (i > 0) {
  //     std::cout << ", ";
  //   }
  //   std::cout << source[i];
  // }
  // std::cout << std::endl;
  // std::cout << "source (last 20): ";
  // for (size_t i = source.size() - 20; i < source.size(); i += 1) {
  //   if (i > 0) {
  //     std::cout << ", ";
  //   }
  //   std::cout << source[i];
  // }
  // std::cout << std::endl;

  // sgpp::base::DataVector temp(storage.getSize());
  // sgpp::datadriven::SubspaceAutoTuneTMP::detail::unflatten(
  //     dim, allSubspaceNodes, allLevelsIndexMap, storage, maxLevel, temp);

  // std::cout << "temp.size(): " << temp.size() << std::endl;
  // std::cout << "temp (first 20): ";
  // for (size_t i = 0; i < 20; i += 1) {
  //   if (i > 0) {
  //     std::cout << ", ";
  //   }
  //   std::cout << temp[i];
  // }
  // std::cout << std::endl;
  // std::cout << "temp (last 20): ";
  // for (size_t i = temp.size() - 20; i < temp.size(); i += 1) {
  //   if (i > 0) {
  //     std::cout << ", ";
  //   }
  //   std::cout << temp[i];
  // }
  // std::cout << std::endl;

#pragma omp parallel
  {
    size_t chunk_data_start;
    size_t chunk_data_end;
    sgpp::datadriven::PartitioningTool::getOpenMPPartitionSegment(
        0, paddedDatasetSize, &chunk_data_start, &chunk_data_end,
        SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS);

    size_t totalThreadNumber = SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS +
                               SUBSPACEAUTOTUNETMP_VEC_PADDING;

    double *evalIndexValuesAll = new double[(dim + 1) * totalThreadNumber];

    for (size_t i = 0; i < (dim + 1) * totalThreadNumber; i++) {
      evalIndexValuesAll[i] = 1.0;
    }

    // for faster index flattening
    uint32_t *intermediatesAll = new uint32_t[(dim + 1) * totalThreadNumber];

    for (size_t i = 0; i < (dim + 1) * totalThreadNumber; i++) {
      intermediatesAll[i] = 0.0;
    }

    size_t validIndices[SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS +
                        SUBSPACEAUTOTUNETMP_VEC_PADDING];
    size_t validIndicesCount;

    size_t levelIndices[SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS +
                        SUBSPACEAUTOTUNETMP_VEC_PADDING];
    // size_t
    // nextIterationToRecalcReferences[SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS
    // + SUBSPACEAUTOTUNETMP_VEC_PADDING];

    double *listSubspace = new double[maxGridPointsOnLevel];

    for (size_t i = 0; i < maxGridPointsOnLevel; i++) {
      listSubspace[i] = std::numeric_limits<double>::quiet_NaN();
    }

    /*uint64_t jumpCount = 0;
    uint64_t jumpDistance = 0;
    uint64_t evaluationCounter = 0;
    uint64_t recomputeDimsTotal = 0;
    vector<uint64_t> dimRecalc(dim, 0);*/

    for (size_t dataIndexBase = chunk_data_start;
         dataIndexBase < chunk_data_end;
         dataIndexBase += SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS) {
      for (size_t i = 0; i < totalThreadNumber; i++) {
        levelIndices[i] = 0.0;
        // nextIterationToRecalcReferences[i] = 0;
      }

      for (size_t subspaceIndex = 0;
           subspaceIndex < allSubspaceNodes.size() - 1; subspaceIndex++) {
        SubspaceNode &subspace = allSubspaceNodes[subspaceIndex];

        // prepare the subspace array for a list type subspace
        if (subspace.type == SubspaceNode::SubspaceType::LIST) {
          // std::cout << "subspace type is LIST" << std::endl;
          // fill with surplusses
          for (std::pair<uint32_t, double> tuple :
               subspace.indexFlatSurplusPairs) {
            // accumulator that are later added to the global surplusses
            listSubspace[tuple.first] = 0.0;
          }
        }

        validIndicesCount = 0;

        for (size_t parallelIndex = 0;
             parallelIndex < SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS;
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
                     SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS +
                         SUBSPACEAUTOTUNETMP_VEC_PADDING);

        for (size_t i = validIndicesCount; i < paddingSize; i++) {
          size_t threadId = SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS +
                            (i - validIndicesCount);
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

          detail::listMultTransposeInner(
              isModLinear, paddedDataset, paddedDatasetSize, dim, source,
              dataIndexBase, chunk_data_end, subspace,
              subspace.subspaceArray.data(), validIndicesCount, validIndices,
              levelIndices, evalIndexValuesAll, intermediatesAll);

          // unlocks the subspace lock for ARRAY and BLUEPRINT type subspaces
          subspace.unlockSubspace();

        } else if (subspace.type == SubspaceNode::SubspaceType::LIST) {
          detail::listMultTransposeInner(
              isModLinear, paddedDataset, paddedDatasetSize, dim, source,
              dataIndexBase, chunk_data_end, subspace, listSubspace,
              validIndicesCount, validIndices, levelIndices, evalIndexValuesAll,
              intermediatesAll);

          // write results into the global surplus array
          // if (subspace.type == SubspaceNode::SubspaceType::LIST) {
          for (std::pair<uint32_t, double> &tuple :
               subspace.indexFlatSurplusPairs) {
            if (listSubspace[tuple.first] != 0.0) {
#pragma omp atomic
              tuple.second += listSubspace[tuple.first];
            }

            listSubspace[tuple.first] =
                std::numeric_limits<double>::quiet_NaN();
            // }
          }
        }
      } // end iterate subspaces
    }   // end iterate chunks

    delete[] evalIndexValuesAll;
    delete[] intermediatesAll;
    delete[] listSubspace;
  }

  sgpp::base::DataVector result(storage.getSize());
  sgpp::datadriven::SubspaceAutoTuneTMP::detail::unflatten(
      dim, allSubspaceNodes, allLevelsIndexMap, storage, maxLevel, result);

  // std::cout << "result.size(): " << result.size() << std::endl;
  // std::cout << "result (first 20): ";
  // for (size_t i = 0; i < 20; i += 1) {
  //   if (i > 0) {
  //     std::cout << ", ";
  //   }
  //   std::cout << result[i];
  // }
  // std::cout << std::endl;
  // std::cout << "result (last 20): ";
  // for (size_t i = result.size() - 20; i < result.size(); i += 1) {
  //   if (i > 0) {
  //     std::cout << ", ";
  //   }
  //   std::cout << result[i];
  // }
  // std::cout << std::endl;

  return result;
}
