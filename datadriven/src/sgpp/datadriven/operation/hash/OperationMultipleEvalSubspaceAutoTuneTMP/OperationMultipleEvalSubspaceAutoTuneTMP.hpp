// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#ifdef __AVX__

#pragma once

#include <assert.h>
#include <immintrin.h>
#include <iostream>
#include <map>
#include <sgpp/base/grid/Grid.hpp>
#include <sgpp/base/grid/GridStorage.hpp>
#include <sgpp/base/operation/hash/OperationMultipleEval.hpp>
#include <sgpp/base/tools/SGppStopwatch.hpp>
#include <sgpp/datadriven/tools/PartitioningTool.hpp>
#include <vector>
#include "SubspaceAutoTuneTMPParameters.hpp"
#include "SubspaceNodeCombined.hpp"
#include "omp.h"

namespace sgpp::datadriven::SubspaceAutoTuneTMP {

/**
 * Multiple evaluation operation that uses the subspace structure to save work
 * compared to the naive or streaming variants.
 */
class OperationMultipleEvalSubspaceAutoTuneTMP : public base::OperationMultipleEval {
 private:
  base::GridStorage &storage;

  base::SGppStopwatch timer;
  double duration;

  sgpp::base::DataMatrix paddedDataset;
  // includes padding, but excludes additional rows for SUBSPACEAUTOTUNETMP_VEC_PADDING
  size_t paddedDatasetSize;

  // size_t subspaceSize = -1;

  size_t maxGridPointsOnLevel;

  std::map<uint32_t, uint32_t> allLevelsIndexMap;

  size_t dim;       // = -1;
  size_t maxLevel;  // = 0;

  std::vector<SubspaceNodeCombined> allSubspaceNodes;
  uint32_t subspaceCount;  // = -1;

  uint32_t totalRegularGridPoints;  // = -1;

  bool isModLinear;

#ifdef SUBSPACEAUTOTUNETMP_WRITE_STATS
  size_t refinementStep = 0;
  ofstream statsFile;
  string csvSep = "& ";
#endif

  /**
   * Creates the data structure used by the operation.
   */
  void prepareSubspaceIterator();

  void setCoefficients(sgpp::base::DataVector &surplusVector);

  void unflatten(sgpp::base::DataVector &result);

  // static uint32_t flattenIndex(size_t dim, std::vector<uint32_t> &maxIndices,
  //                              std::vector<uint32_t> &index);

  static inline uint32_t flattenIndex(const size_t dim, const std::vector<uint32_t> &maxIndices,
                                      const std::vector<uint32_t> &index) {
    uint32_t indexFlat = index[0];
    indexFlat >>= 1;

    for (size_t i = 1; i < dim; i++) {
      uint32_t actualDirectionGridPoints = maxIndices[i];
      actualDirectionGridPoints >>= 1;
      indexFlat *= actualDirectionGridPoints;
      uint32_t actualIndex = index[i];
      actualIndex >>= 1;  // divide index by 2, skip even indices
      indexFlat += actualIndex;
    }

    return indexFlat;
  }

  void setSurplus(std::vector<uint32_t> &level, std::vector<uint32_t> &maxIndices,
                  std::vector<uint32_t> &index, double value);

  void getSurplus(std::vector<uint32_t> &level, std::vector<uint32_t> &maxIndices,
                  std::vector<uint32_t> &index, double &value, bool &isVirtual);

  uint32_t flattenLevel(size_t dim, size_t maxLevel, std::vector<uint32_t> &level);

 public:
  /**
   * Creates a new instance of the OperationMultipleEvalSubspaceAutoTuneTMP class.
   *
   * @param grid grid to be evaluated
   * @param dataset set of evaluation points
   */
  OperationMultipleEvalSubspaceAutoTuneTMP(sgpp::base::Grid &grid, sgpp::base::DataMatrix &dataset,
                                           bool isModLinear);

  /**
   * Destructor
   */
  ~OperationMultipleEvalSubspaceAutoTuneTMP();

  void multTranspose(sgpp::base::DataVector &alpha, sgpp::base::DataVector &result) override {
    if (!this->isPrepared) {
      this->prepare();
    }

    size_t originalAlphaSize = alpha.getSize();

    const size_t start_index_data = 0;
    const size_t end_index_data = this->getPaddedDatasetSize();

    // pad the alpha vector to the padded size of the dataset
    alpha.resizeZero(this->getPaddedDatasetSize());

    this->timer.start();
    result.setAll(0.0);

#pragma omp parallel
    {
      size_t start;
      size_t end;
      PartitioningTool::getOpenMPPartitionSegment(start_index_data, end_index_data, &start, &end,
                                                  this->getAlignment());
      this->multTransposeImpl(alpha, result, start, end);
    }

    alpha.resize(originalAlphaSize);
    this->duration = this->timer.stop();
  }

  void mult(sgpp::base::DataVector &source, sgpp::base::DataVector &result) override {
    if (!this->isPrepared) {
      this->prepare();
    }

    size_t originalResultSize = result.getSize();
    result.resizeZero(this->getPaddedDatasetSize());

    const size_t start_index_data = 0;
    const size_t end_index_data = this->getPaddedDatasetSize();

    this->timer.start();
    result.setAll(0.0);

#pragma omp parallel
    {
      size_t start;
      size_t end;
      PartitioningTool::getOpenMPPartitionSegment(start_index_data, end_index_data, &start, &end,
                                                  this->getAlignment());
      this->multImpl(source, result, start, end);
    }

    result.resize(originalResultSize);

    this->duration = this->timer.stop();
  }

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
  void multTransposeImpl(sgpp::base::DataVector &alpha, sgpp::base::DataVector &result,
                         const size_t start_index_data, const size_t end_index_data);

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
                const size_t start_index_data, const size_t end_index_data);

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
  size_t getAlignment();

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
  size_t getPaddedDatasetSize();

  virtual double getDuration() override { return this->duration; }

  static inline size_t getChunkGridPoints() { return 12; }
  static inline size_t getChunkDataPoints() {
    return 24;  // must be divisible by 24
  }
};

}  // namespace sgpp::datadriven::SubspaceAutoTuneTMP

#endif
