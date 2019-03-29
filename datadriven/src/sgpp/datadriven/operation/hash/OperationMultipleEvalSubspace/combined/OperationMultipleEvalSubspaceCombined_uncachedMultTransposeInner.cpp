// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#include "../../OperationMultipleEvalSubspace/combined/OperationMultipleEvalSubspaceCombined.hpp"

#include <sgpp/globaldef.hpp>

namespace sgpp::datadriven::SubspaceLinearCombined {

void OperationMultipleEvalSubspaceCombined::uncachedMultTransposeInner(
    size_t curDataStart, SubspaceNodeCombined &subspace,
    double *curSubspaceSurpluses, size_t validIndicesCount,
    std::array<size_t, X86COMBINED_PARALLEL_DATA_POINTS +
                           X86COMBINED_VEC_PADDING> &validIndices,
    std::array<size_t, X86COMBINED_PARALLEL_DATA_POINTS +
                           X86COMBINED_VEC_PADDING> &nextSubspaceIndex,
    std::array<double, X86COMBINED_PARALLEL_DATA_POINTS +
                           X86COMBINED_VEC_PADDING> &componentResults,
    std::vector<double> &partialPhiEvalsSchedule,
    std::vector<uint32_t> &partialIndicesFlatSchedule) {
  // iterate the indices of the datapoints that evaluate on this subspace
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

#if X86COMBINED_UNROLL == 1
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

    OperationMultipleEvalSubspaceCombined::calculateIndexCombined2(
        isModLinear, dim, nextIterationToRecalc, dataTuplePtr, dataTuplePtr2,
        subspace.hInverse, partialIndicesFlat, partialIndicesFlat2,
        evalIndexValues, evalIndexValues2, indexFlat, indexFlat2, phiEval,
        phiEval2);
#else
    OperationMultipleEvalSubspaceCombined::calculateIndexCombined(
        isModLinear, dim, nextIterationToRecalc, dataTuplePtr,
        subspace.hInverse, partialIndicesFlat, evalIndexValues, indexFlat,
        phiEval);
#endif

    double surplus[4];
    surplus[0] = curSubspaceSurpluses[indexFlat[0]];
    surplus[1] = curSubspaceSurpluses[indexFlat[1]];
    surplus[2] = curSubspaceSurpluses[indexFlat[2]];
    surplus[3] = curSubspaceSurpluses[indexFlat[3]];

#if X86COMBINED_UNROLL == 1
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
#if X86COMBINED_ENABLE_SUBSPACE_SKIPPING == 1
        // skip to next relevant subspace
        nextSubspaceIndex[validIndices[validIndex + innerIndex]] =
            subspace.jumpTargetIndex;
#else
        nextSubspaceIndex[validIndices[validIndex + innerIndex]] += 1;
#endif
      }
    }

#if X86COMBINED_UNROLL == 1
    for (size_t innerIndex = 0; innerIndex < 4; innerIndex++) {
      if (!std::isnan(surplus2[innerIndex])) {
        componentResults[validIndices[validIndex + innerIndex + 4]] +=
            phiEval2[innerIndex] * surplus2[innerIndex];
        nextSubspaceIndex[validIndices[validIndex + innerIndex + 4]] += 1;
      } else {
#if X86COMBINED_ENABLE_SUBSPACE_SKIPPING == 1
        // skip to next relevant subspace
        nextSubspaceIndex[validIndices[validIndex + innerIndex + 4]] =
            subspace.jumpTargetIndex;
#else
        nextSubspaceIndex[validIndices[validIndex + innerIndex + 4]] += 1;
#endif
      }
    }

#endif
  } // end X86COMBINED_PARALLEL_DATA_POINTS
}

} // namespace sgpp::datadriven::SubspaceLinearCombined
