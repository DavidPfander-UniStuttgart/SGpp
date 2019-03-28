// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#ifdef __AVX__

#pragma once

#include "omp.h"
#include <assert.h>
#include <immintrin.h>
#include <iostream>
#include <map>
#include <vector>

#include "OperationMultipleEvalSubspaceCombinedParameters.hpp"
#include "SubspaceNodeCombined.hpp"
#include <sgpp/datadriven/operation/hash/OperationMultipleEvalSubspace/AbstractOperationMultipleEvalSubspace.hpp>

#include <sgpp/globaldef.hpp>

namespace sgpp::datadriven::SubspaceLinearCombined {

/**
 * Multiple evaluation operation that uses the subspace structure to save work
 * compared to the naive or streaming variants.
 */
class OperationMultipleEvalSubspaceCombined
    : public AbstractOperationMultipleEvalSubspace {
private:
  sgpp::base::DataMatrix paddedDataset;
  // includes padding, but excludes additional rows for X86COMBINED_VEC_PADDING
  size_t paddedDatasetSize;

  // size_t subspaceSize = -1;

  size_t maxGridPointsOnLevel;

  std::map<uint32_t, uint32_t> allLevelsIndexMap;

  size_t dim;      // = -1;
  size_t maxLevel; // = 0;

  std::vector<SubspaceNodeCombined> allSubspaceNodes;
  uint32_t subspaceCount; // = -1;

  uint32_t totalRegularGridPoints; // = -1;

#ifdef X86COMBINED_WRITE_STATS
  size_t refinementStep = 0;
  ofstream statsFile;
  string csvSep = "& ";
#endif

  /**
   * Creates the data structure used by the operation.
   */
  void prepareSubspaceIterator();

  void listMultInner(
      size_t dim, sgpp::base::DataVector &alpha, size_t dataIndexBase,
      size_t end_index_data, SubspaceNodeCombined &subspace,
      double *levelArrayContinuous, size_t validIndicesCount,
      size_t *validIndices, size_t *levelIndices,
      // size_t *nextIterationToRecalcReferences, size_t nextIterationToRecalc,
      double *evalIndexValuesAll, uint32_t *intermediatesAll);

  void uncachedMultTransposeInner(
      size_t curDataStart, SubspaceNodeCombined &subspace,
      double *curSubspaceSurpluses, size_t validIndicesCount,
      std::array<size_t, X86COMBINED_PARALLEL_DATA_POINTS +
                             X86COMBINED_VEC_PADDING> &validIndices,
      std::array<size_t, X86COMBINED_PARALLEL_DATA_POINTS +
                             X86COMBINED_VEC_PADDING> &nextSubspaceIndex,
      std::array<double, X86COMBINED_PARALLEL_DATA_POINTS +
                             X86COMBINED_VEC_PADDING> &componentResults,
      std::vector<double> &partialPhiEvalsSchedule,
      std::vector<uint32_t> &intermediatesAll);

  void setCoefficients(sgpp::base::DataVector &surplusVector);

  void unflatten(sgpp::base::DataVector &result);

  // static uint32_t flattenIndex(size_t dim, std::vector<uint32_t> &maxIndices,
  //                              std::vector<uint32_t> &index);

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

  void setSurplus(std::vector<uint32_t> &level,
                  std::vector<uint32_t> &maxIndices,
                  std::vector<uint32_t> &index, double value);

  void getSurplus(std::vector<uint32_t> &level,
                  std::vector<uint32_t> &maxIndices,
                  std::vector<uint32_t> &index, double &value, bool &isVirtual);

  uint32_t flattenLevel(size_t dim, size_t maxLevel,
                        std::vector<uint32_t> &level);

public:
#include "OperationMultipleEvalSubspaceCombined_calculateIndexCombined.hpp"

  /**
   * Creates a new instance of the OperationMultipleEvalSubspaceCombined class.
   *
   * @param grid grid to be evaluated
   * @param dataset set of evaluation points
   */
  OperationMultipleEvalSubspaceCombined(sgpp::base::Grid &grid,
                                        sgpp::base::DataMatrix &dataset);

  /**
   * Destructor
   */
  ~OperationMultipleEvalSubspaceCombined();

  /**
   * Updates the internal data structures to reflect changes to the grid, e.g.
   * due to refinement.
   *
   */
  void prepare() override;

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
  void multTransposeImpl(sgpp::base::DataVector &alpha,
                         sgpp::base::DataVector &result,
                         const size_t start_index_data,
                         const size_t end_index_data) override;

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
  void multImpl(sgpp::base::DataVector &source, sgpp::base::DataVector &result,
                const size_t start_index_data,
                const size_t end_index_data) override;

  /**
   * Pads the dataset.
   *
   * @param dataset dataset to be padded
   */
  void padDataset(sgpp::base::DataMatrix &dataset);

  /**
   * Alignment required by the vector instruction set SG++ is compiled with.
   *
   * @result alignment requirement
   */
  size_t getAlignment() override;

  /**
   * Name of the implementation, useful for benchmarking different
   * implementation approaches.
   *
   * @result name of the implementation
   */
  std::string getImplementationName() override;

  /**
   * Size of the dataset after padding.
   *
   * @result size of the padded dataset>
   */
  size_t getPaddedDatasetSize() override;
};

} // namespace sgpp::datadriven::SubspaceLinearCombined

#endif
