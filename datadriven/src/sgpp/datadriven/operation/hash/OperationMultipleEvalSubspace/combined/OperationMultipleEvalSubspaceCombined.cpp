// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#include "OperationMultipleEvalSubspaceCombined.hpp"
#include <sgpp/datadriven/operation/hash/OperationMultipleEvalSubspace/AbstractOperationMultipleEvalSubspace.hpp>

#include <sgpp/globaldef.hpp>

#include <string>
#include <vector>

using sgpp::base::DataMatrix;
using sgpp::base::DataVector;
using sgpp::base::Grid;

namespace sgpp::datadriven::SubspaceLinearCombined {

OperationMultipleEvalSubspaceCombined::OperationMultipleEvalSubspaceCombined(
    Grid &grid, DataMatrix &dataset)
    : AbstractOperationMultipleEvalSubspace(grid, dataset),
      paddedDatasetSize(0), maxGridPointsOnLevel(0), dim(dataset.getNcols()),
      maxLevel(0), subspaceCount(-1), totalRegularGridPoints(-1) {
  this->padDataset(dataset);

#ifdef X86COMBINED_WRITE_STATS
  string prefix("results/data/stats_");
  string fileName(X86COMBINED_WRITE_STATS);
  this->statsFile.open(prefix + fileName, ios::out);

  this->statsFile << "# name: " << X86COMBINED_WRITE_STATS_NAME << endl;
  this->statsFile << "refinementStep & ";
  this->statsFile << "nonVirtualGridPoints & ";
  this->statsFile << "totalRegularGridPoints & ";
  this->statsFile << "actualGridPoints & ";
  this->statsFile << "largestArraySubspace & ";
  this->statsFile << "largestListSubspace & ";
  this->statsFile << "numberOfListSubspaces & ";
  this->statsFile << "subspaceCount & ";
  this->statsFile << "avrPointsPerSubspace & ";
  this->statsFile << "memoryEstimate & ";
  this->statsFile << "memoryEfficiency";
  this->statsFile << endl;

#endif
}

OperationMultipleEvalSubspaceCombined::
    ~OperationMultipleEvalSubspaceCombined() {
#ifdef X86COMBINED_WRITE_STATS
  this->statsFile.close();
#endif
}

void OperationMultipleEvalSubspaceCombined::prepare() {
  this->prepareSubspaceIterator();
}

void OperationMultipleEvalSubspaceCombined::setCoefficients(
    DataVector &surplusVector) {
  std::vector<uint32_t> level(dim);
  std::vector<uint32_t> maxIndex(dim);
  std::vector<uint32_t> index(dim);

  base::level_t curLevel;
  base::index_t curIndex;

  for (size_t gridPoint = 0; gridPoint < this->storage.getSize(); gridPoint++) {
    sgpp::base::GridPoint &point = this->storage.getPoint(gridPoint);

    for (size_t d = 0; d < this->dim; d++) {
      point.get(d, curLevel, curIndex);
      level[d] = curLevel;
      index[d] = curIndex;
      maxIndex[d] = 1 << curLevel;
    }

    this->setSurplus(level, maxIndex, index, surplusVector.get(gridPoint));
  }
}

// writes a result vector in the order of the points in the grid storage
void OperationMultipleEvalSubspaceCombined::unflatten(DataVector &result) {
  std::vector<uint32_t> level(dim);
  std::vector<uint32_t> maxIndex(dim);
  std::vector<uint32_t> index(dim);

  base::level_t curLevel;
  base::index_t curIndex;

  for (size_t gridPoint = 0; gridPoint < this->storage.getSize(); gridPoint++) {
    sgpp::base::GridPoint &point = this->storage.getPoint(gridPoint);

    for (size_t d = 0; d < this->dim; d++) {
      point.get(d, curLevel, curIndex);
      level[d] = curLevel;
      index[d] = curIndex;
      maxIndex[d] = 1 << curLevel;
    }

    double surplus;
    bool isVirtual;
    this->getSurplus(level, maxIndex, index, surplus, isVirtual);

    result.set(gridPoint, surplus);
  }
}

void OperationMultipleEvalSubspaceCombined::setSurplus(
    std::vector<uint32_t> &level, std::vector<uint32_t> &maxIndices,
    std::vector<uint32_t> &index, double value) {
  uint32_t levelFlat = this->flattenLevel(this->dim, this->maxLevel, level);
  uint32_t indexFlat = this->flattenIndex(this->dim, maxIndices, index);
  uint32_t subspaceIndex = this->allLevelsIndexMap.find(levelFlat)->second;
  SubspaceNodeCombined &subspace = this->allSubspaceNodes[subspaceIndex];
  subspace.setSurplus(indexFlat, value);
}

void OperationMultipleEvalSubspaceCombined::getSurplus(
    std::vector<uint32_t> &level, std::vector<uint32_t> &maxIndices,
    std::vector<uint32_t> &index, double &value, bool &isVirtual) {
  uint32_t levelFlat = this->flattenLevel(this->dim, this->maxLevel, level);
  uint32_t indexFlat = this->flattenIndex(this->dim, maxIndices, index);
  uint32_t subspaceIndex = this->allLevelsIndexMap.find(levelFlat)->second;
  SubspaceNodeCombined &subspace = this->allSubspaceNodes[subspaceIndex];
  value = subspace.getSurplus(indexFlat);

  if (std::isnan(value)) {
    isVirtual = true;
  } else {
    isVirtual = false;
  }
}

uint32_t OperationMultipleEvalSubspaceCombined::flattenLevel(
    size_t dim, size_t maxLevel, std::vector<uint32_t> &level) {
  uint32_t levelFlat = 0;
  levelFlat += level[dim - 1];

  // loop terminates at -1
  for (int i = static_cast<int>(dim - 2); i >= 0; i--) {
    levelFlat *= static_cast<uint32_t>(maxLevel);
    levelFlat += level[i];
  }

  return levelFlat;
}

void OperationMultipleEvalSubspaceCombined::padDataset(
    sgpp::base::DataMatrix &dataset) {
  size_t chunkSize = X86COMBINED_PARALLEL_DATA_POINTS;

  // Assure that data has a even number of instances -> padding might be needed
  size_t remainder = dataset.getNrows() % chunkSize;
  size_t loopCount = chunkSize - remainder;

  // if (loopCount == chunkSize) {
  //   std::endl; return &dataset;
  // }

  paddedDataset = DataMatrix(dataset);

  // due to rounding issue in calculateIndex, replace all values of 1 by
  // calculating the index of the grid point for a given level and data point (in 1d) treats the right border as part of the next grid point (ascending). This leads incorrect values for the right-most grid points.
  double one = 1.0;
  // subtract to obtain the next smaller float
  uint64_t temp = reinterpret_cast<uint64_t &>(one) - 1;
  double replace_value = reinterpret_cast<double &>(temp);
  for (size_t i = 0; i < paddedDataset.size(); i += 1) {
    if (paddedDataset[i] == 1.0) {
      paddedDataset[i] = replace_value;
    }
  }
  // std::cout.precision(17);
  // std::cout << "one: " << one << " replace: " << replace_value << std::endl;

  // pad to make: dataset % X86COMBINED_PARALLEL_DATA_POINTS == 0
  if (loopCount != chunkSize) {
    sgpp::base::DataVector lastRow(dataset.getNcols());
    size_t oldSize = dataset.getNrows();
    dataset.getRow(dataset.getNrows() - 1, lastRow);
    paddedDataset.resize(dataset.getNrows() + loopCount);

    for (size_t i = 0; i < loopCount; i++) {
      paddedDataset.setRow(oldSize + i, lastRow);
    }
  }
  paddedDatasetSize = paddedDataset.getNrows();

  // TODO: in the process of changing this, as the old approach is bullshit
  // (accessing reserved parts of a vector...) additional padding for subspace
  // skipping if validIndices contain X86COMBINED_PARALLEL_DATA_POINTS - 1 it is
  // possible for a vector iteration to contain indices larger than
  // size(dataset) (even though the dataset is divided by
  // X86COMBINED_PARALLEL_DATA_POINTS) add X86COMBINED_VEC_PADDING dummy data
  // points to avoid that problem add X86COMBINED_VEC_PADDING * 2 to also enable
  // the calculateIndexCombined2() method this works due to special semantics of
  // "reserveAdditionalRows()", this function adds additional unused (and
  // uncounted) rows
  // paddedDataset->reserveAdditionalRows(X86COMBINED_VEC_PADDING * 2);

  paddedDataset.resize(paddedDataset.size() +
                       X86COMBINED_VEC_PADDING * 2 * paddedDataset.getNcols());

  for (size_t i = paddedDatasetSize; i < paddedDataset.getNrows(); i += 1) {
    for (size_t j = 0; j < paddedDataset.getNcols(); j += 1) {
      paddedDataset.set(i, j, 0.0);
    }
  }
}

size_t OperationMultipleEvalSubspaceCombined::getPaddedDatasetSize() {
  return paddedDatasetSize;
}

size_t OperationMultipleEvalSubspaceCombined::getAlignment() {
  return X86COMBINED_PARALLEL_DATA_POINTS;
}

std::string OperationMultipleEvalSubspaceCombined::getImplementationName() {
  return "COMBINED";
}

} // namespace sgpp::datadriven::SubspaceLinearCombined
