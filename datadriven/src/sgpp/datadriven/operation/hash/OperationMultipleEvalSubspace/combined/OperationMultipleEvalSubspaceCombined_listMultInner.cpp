// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#include "OperationMultipleEvalSubspaceCombined.hpp"

#include <sgpp/globaldef.hpp>

namespace sgpp::datadriven::SubspaceLinearCombined {

void OperationMultipleEvalSubspaceCombined::listMultInner(
    size_t dim, sgpp::base::DataVector &alpha, size_t dataIndexBase,
    size_t end_index_data, SubspaceNodeCombined &subspace,
    double *levelArrayContinuous, size_t validIndicesCount,
    size_t *validIndices, size_t *levelIndices, double *evalIndexValuesAll,
    uint32_t *intermediatesAll) {
  for (size_t validIndex = 0; validIndex < validIndicesCount;
       validIndex += X86COMBINED_VEC_PADDING) {
    size_t parallelIndices[4];
    parallelIndices[0] = validIndices[validIndex];
    parallelIndices[1] = validIndices[validIndex + 1];
    parallelIndices[2] = validIndices[validIndex + 2];
    parallelIndices[3] = validIndices[validIndex + 3];

#if X86COMBINED_ENABLE_PARTIAL_RESULT_REUSAGE == 1
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

#if X86COMBINED_UNROLL == 1
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

    OperationMultipleEvalSubspaceCombined::calculateIndexCombined2(
        dim, nextIterationToRecalc, dataTuplePtr, dataTuplePtr2,
        subspace.hInverse, intermediates, intermediates2, evalIndexValues,
        evalIndexValues2, indexFlat, indexFlat2, phiEval, phiEval2);
#else
    OperationMultipleEvalSubspaceCombined::calculateIndexCombined(
        dim, nextIterationToRecalc, dataTuplePtr, subspace.hInverse,
        intermediates, evalIndexValues, indexFlat, phiEval);
#endif

    double surplus[4];
    if (indexFlat[0] >= subspace.gridPointsOnLevel) {
      std::cout << "indexFlat 0 out of bounds: " << indexFlat[0] << std::endl;
      throw;
    }
    if (indexFlat[1] >= subspace.gridPointsOnLevel) {
      std::cout << "indexFlat 1 out of bounds: " << indexFlat[1] << std::endl;
      throw;
    }
    if (indexFlat[2] >= subspace.gridPointsOnLevel) {
      std::cout << "indexFlat 2 out of bounds: " << indexFlat[2] << std::endl;
      throw;
    }
    if (indexFlat[3] >= subspace.gridPointsOnLevel) {
      std::cout << "indexFlat 3 out of bounds: " << indexFlat[3] << std::endl;
      throw;
    }

    // std::cout << "indexFlat 0: " << indexFlat[0]
    //           << "indexFlat 1: " << indexFlat[1]
    //           << "indexFlat 2: " << indexFlat[2]
    //           << "indexFlat 3: " << indexFlat[3] << std::endl;
    surplus[0] = levelArrayContinuous[indexFlat[0]];
    surplus[1] = levelArrayContinuous[indexFlat[1]];
    surplus[2] = levelArrayContinuous[indexFlat[2]];
    surplus[3] = levelArrayContinuous[indexFlat[3]];

#if X86COMBINED_UNROLL == 1
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
            parallelIndex < X86COMBINED_PARALLEL_DATA_POINTS) {
          partialSurplus =
              phiEval[innerIndex] * alpha[dataIndexBase + parallelIndex];

          size_t localIndexFlat = indexFlat[innerIndex];

          // no atomics required, working on temporary arrays
          //#pragma omp atomic
          levelArrayContinuous[localIndexFlat] += partialSurplus;
        }

        // nextIterationToRecalcReferences[parallelIndex] = subspace.nextDiff;
        levelIndices[parallelIndex] += 1;
      } else {
#if X86COMBINED_ENABLE_SUBSPACE_SKIPPING == 1
        // skip to next relevant subspace
        // nextIterationToRecalcReferences[parallelIndex] = subspace.jumpDiff;
        levelIndices[parallelIndex] = subspace.jumpTargetIndex;
#else
        // nextIterationToRecalcReferences[parallelIndex] = subspace.nextDiff;
        levelIndices[parallelIndex] += 1;
#endif
      }
    } // end innerIndex

#if X86COMBINED_UNROLL == 1

    // for second vector
    for (size_t innerIndex = 0; innerIndex < 4; innerIndex++) {
      size_t parallelIndex = parallelIndices2[innerIndex];

      if (!std::isnan(surplus2[innerIndex])) {
        double partialSurplus = 0.0;

        if (dataIndexBase + parallelIndex < end_index_data &&
            parallelIndex < X86COMBINED_PARALLEL_DATA_POINTS) {
          partialSurplus =
              phiEval2[innerIndex] * alpha[dataIndexBase + parallelIndex];

          size_t localIndexFlat = indexFlat2[innerIndex];

          // no atomics required, subspace is locked before processing
          //#pragma omp atomic
          levelArrayContinuous[localIndexFlat] += partialSurplus;
        }

        // nextIterationToRecalcReferences[parallelIndex] = subspace.nextDiff;
        levelIndices[parallelIndex] += 1;
      } else {
#if X86COMBINED_ENABLE_SUBSPACE_SKIPPING == 1
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

} // namespace sgpp::datadriven::SubspaceLinearCombined
