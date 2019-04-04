// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#ifdef __AVX__

#pragma once

#include "SubspaceNode.hpp"
#include "omp.h"
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

namespace sgpp::datadriven::SubspaceAutoTuneTMP {

/**
 * Multiple evaluation operation that uses the subspace structure to save work
 * compared to the naive or streaming variants.
 */
class OperationMultipleEvalSubspaceAutoTuneTMP
    : public base::OperationMultipleEval {
private:
  base::GridStorage &storage;
  base::SGppStopwatch timer;
  double duration;
  sgpp::base::DataMatrix paddedDataset;
  // includes padding, but excludes additional rows for
  // "vectorPadding"
  size_t paddedDatasetSize;
  size_t maxGridPointsOnLevel;
  std::map<uint32_t, uint32_t> allLevelsIndexMap;
  size_t dim;
  size_t maxLevel;
  std::vector<SubspaceNode> allSubspaceNodes;
  uint32_t subspaceCount;
  uint32_t totalRegularGridPoints;
  bool isModLinear;
  size_t refinementStep;
  std::ofstream statsFile;
  std::string csvSep;
  bool write_stats;

  double listRatio;
  // if below this threshold (in number of grid points), subspace is streamed
  int64_t streamingThreshold;
  // chunk of data points processed by a single thread, needs to devide
  // vector-variable size (i.e. 4 for AVX)
  int64_t parallelDataPoints;
  // padding for chunk ("parallelDataPoints"), should be set to size of vector
  // type, unroll requires 2*vector-size
  int64_t vectorPadding;

  const uint64_t AVX_vector_width = 4;

  /**
   * Creates the data structure used by the operation.
   */
  void prepareSubspaceIterator();

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

  void setCoefficients(base::DataVector &surplusVector);

public:
  /**
   * Creates a new instance of the OperationMultipleEvalSubspaceAutoTuneTMP
   * class.
   *
   * @param grid grid to be evaluated
   * @param dataset set of evaluation points
   */
  OperationMultipleEvalSubspaceAutoTuneTMP(sgpp::base::Grid &grid,
                                           sgpp::base::DataMatrix &dataset,
                                           bool isModLinear);

  /**
   * Destructor
   */
  ~OperationMultipleEvalSubspaceAutoTuneTMP();

  void multTranspose(sgpp::base::DataVector &source,
                     sgpp::base::DataVector &result) override;

  void mult(sgpp::base::DataVector &alpha,
            sgpp::base::DataVector &result) override;

  /**
   * Updates the internal data structures to reflect changes to the grid, e.g.
   * due to refinement.
   *
   */
  void prepare() override;

  /**
   * Pads the dataset.
   *
   * @param dataset dataset to be padded
   */
  void padDataset(sgpp::base::DataMatrix &dataset);

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

  virtual double getDuration() override;

  static inline size_t getChunkGridPoints() { return 12; }
  static inline size_t getChunkDataPoints() {
    return 24; // must be divisible by 24
  }

  void set_write_stats(const std::string &stats_file_name,
                       const std::string &name = "");

  void tune_mult(sgpp::base::DataVector &alpha, sgpp::base::DataVector &result,
                 const std::string &scenario_name,
                 const std::string &tuner_name, uint32_t repetitions);

  void tune_multTranspose(sgpp::base::DataVector &source,
                          sgpp::base::DataVector &result,
                          const std::string &scenario_name,
                          const std::string &tuner_name, uint32_t repetitions);
};

} // namespace sgpp::datadriven::SubspaceAutoTuneTMP

#endif
