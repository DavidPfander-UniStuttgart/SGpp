// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#include "OperationMultipleEvalSubspaceAutoTuneTMP.hpp"
#include <string>
#include <vector>
#include "autotune/autotune.hpp"

using sgpp::base::DataMatrix;
using sgpp::base::DataVector;
using sgpp::base::Grid;

// void multImplSubspace(
//     size_t maxGridPointsOnLevel, bool isModLinear, sgpp::base::DataMatrix &paddedDataset,
//     size_t paddedDatasetSize,
//     std::vector<sgpp::datadriven::SubspaceAutoTuneTMP::SubspaceNode> &allSubspaceNodes,
//     sgpp::base::DataVector &source, sgpp::base::DataVector &result, const size_t
//     start_index_data,
//     const size_t end_index_data);

AUTOTUNE_KERNEL(void(size_t, bool, sgpp::base::DataMatrix &, size_t,
                     std::vector<sgpp::datadriven::SubspaceAutoTuneTMP::SubspaceNode> &,
                     sgpp::base::DataVector &, sgpp::base::DataVector &, size_t, size_t),
                KernelMultSubspace, "SubspaceAutoTuneTMPKernels/Mult")

// void multTransposeImplSubspace(
//     size_t maxGridPointsOnLevel, bool isModLinear, sgpp::base::DataMatrix &paddedDataset,
//     size_t paddedDatasetSize,
//     std::vector<sgpp::datadriven::SubspaceAutoTuneTMP::SubspaceNode> &allSubspaceNodes,
//     sgpp::base::DataVector &alpha, sgpp::base::DataVector &result, const size_t start_index_data,
//     const size_t end_index_data);

AUTOTUNE_KERNEL(void(size_t, bool, sgpp::base::DataMatrix &, size_t,
                     std::vector<sgpp::datadriven::SubspaceAutoTuneTMP::SubspaceNode> &,
                     sgpp::base::DataVector &, sgpp::base::DataVector &, size_t, size_t),
                KernelMultTransposeSubspace, "SubspaceAutoTuneTMPKernels/MultTranspose")

namespace sgpp::datadriven::SubspaceAutoTuneTMP {

namespace detail {
void get_includes_from_env(std::string &sgpp_base_include, std::string &boost_include,
                           std::string &autotunetmp_include, std::string &vc_include) {
  const char *sgpp_base_include_env = std::getenv("SGPP_BASE_INCLUDE_DIR");
  if (sgpp_base_include_env) {
    sgpp_base_include = std::string("-I") + std::string(sgpp_base_include_env) + std::string(" ");
  } else {
    throw;
  }
  const char *boost_include_env = std::getenv("BOOST_INCLUDE_DIR");
  if (boost_include_env) {
    boost_include = std::string("-I") + std::string(boost_include_env) + std::string(" ");
  } else {
    throw;
  }
  const char *autotunetmp_include_env = std::getenv("AUTOTUNETMP_INCLUDE_DIR");
  if (autotunetmp_include_env) {
    autotunetmp_include =
        std::string("-I") + std::string(autotunetmp_include_env) + std::string(" ");
  } else {
    throw;
  }
  const char *vc_include_env = std::getenv("VC_INCLUDE_DIR");
  if (vc_include_env) {
    vc_include = std::string("-I") + std::string(vc_include_env) + std::string(" ");
  } else {
    throw;
  }
}
}

OperationMultipleEvalSubspaceAutoTuneTMP::OperationMultipleEvalSubspaceAutoTuneTMP(
    Grid &grid, DataMatrix &dataset, bool isModLinear)
    : base::OperationMultipleEval(grid, dataset),
      storage(grid.getStorage()),
      duration(-1.0),
      paddedDatasetSize(0),
      maxGridPointsOnLevel(0),
      dim(dataset.getNcols()),
      maxLevel(0),
      subspaceCount(-1),
      totalRegularGridPoints(-1),
      isModLinear(isModLinear) {
  this->padDataset(dataset);

#ifdef SUBSPACEAUTOTUNETMP_WRITE_STATS
  string prefix("results/data/stats_");
  string fileName(SUBSPACEAUTOTUNETMP_WRITE_STATS);
  this->statsFile.open(prefix + fileName, ios::out);

  this->statsFile << "# name: " << SUBSPACEAUTOTUNETMP_WRITE_STATS_NAME << endl;
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

OperationMultipleEvalSubspaceAutoTuneTMP::~OperationMultipleEvalSubspaceAutoTuneTMP() {
#ifdef SUBSPACEAUTOTUNETMP_WRITE_STATS
  this->statsFile.close();
#endif
}

void OperationMultipleEvalSubspaceAutoTuneTMP::prepare() { this->prepareSubspaceIterator(); }

void OperationMultipleEvalSubspaceAutoTuneTMP::setCoefficients(DataVector &surplusVector) {
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
void OperationMultipleEvalSubspaceAutoTuneTMP::unflatten(DataVector &result) {
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

void OperationMultipleEvalSubspaceAutoTuneTMP::setSurplus(std::vector<uint32_t> &level,
                                                          std::vector<uint32_t> &maxIndices,
                                                          std::vector<uint32_t> &index,
                                                          double value) {
  uint32_t levelFlat = this->flattenLevel(this->dim, this->maxLevel, level);
  uint32_t indexFlat = this->flattenIndex(this->dim, maxIndices, index);
  uint32_t subspaceIndex = this->allLevelsIndexMap.find(levelFlat)->second;
  SubspaceNode &subspace = this->allSubspaceNodes[subspaceIndex];
  subspace.setSurplus(indexFlat, value);
}

void OperationMultipleEvalSubspaceAutoTuneTMP::getSurplus(std::vector<uint32_t> &level,
                                                          std::vector<uint32_t> &maxIndices,
                                                          std::vector<uint32_t> &index,
                                                          double &value, bool &isVirtual) {
  uint32_t levelFlat = this->flattenLevel(this->dim, this->maxLevel, level);
  uint32_t indexFlat = this->flattenIndex(this->dim, maxIndices, index);
  uint32_t subspaceIndex = this->allLevelsIndexMap.find(levelFlat)->second;
  SubspaceNode &subspace = this->allSubspaceNodes[subspaceIndex];
  value = subspace.getSurplus(indexFlat);

  if (std::isnan(value)) {
    isVirtual = true;
  } else {
    isVirtual = false;
  }
}

uint32_t OperationMultipleEvalSubspaceAutoTuneTMP::flattenLevel(size_t dim, size_t maxLevel,
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

void OperationMultipleEvalSubspaceAutoTuneTMP::padDataset(sgpp::base::DataMatrix &dataset) {
  size_t chunkSize = SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS;

  // Assure that data has a even number of instances -> padding might be needed
  size_t remainder = dataset.getNrows() % chunkSize;
  size_t loopCount = chunkSize - remainder;

  // if (loopCount == chunkSize) {
  //   std::endl; return &dataset;
  // }

  paddedDataset = DataMatrix(dataset);

  // due to rounding issue in calculateIndex, replace all values of 1 by
  // calculating the index of the grid point for a given level and data point
  // (in 1d) treats the right border as part of the next grid point (ascending).
  // This leads incorrect values for the right-most grid points.
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

  // pad to make: dataset % SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS == 0
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
  // skipping if validIndices contain SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS - 1 it is
  // possible for a vector iteration to contain indices larger than
  // size(dataset) (even though the dataset is divided by
  // SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS) add SUBSPACEAUTOTUNETMP_VEC_PADDING dummy data
  // points to avoid that problem add SUBSPACEAUTOTUNETMP_VEC_PADDING * 2 to also enable
  // the calculateIndexCombined2() method this works due to special semantics of
  // "reserveAdditionalRows()", this function adds additional unused (and
  // uncounted) rows
  // paddedDataset->reserveAdditionalRows(SUBSPACEAUTOTUNETMP_VEC_PADDING * 2);

  paddedDataset.resize(paddedDataset.size() +
                       SUBSPACEAUTOTUNETMP_VEC_PADDING * 2 * paddedDataset.getNcols());

  for (size_t i = paddedDatasetSize; i < paddedDataset.getNrows(); i += 1) {
    for (size_t j = 0; j < paddedDataset.getNcols(); j += 1) {
      paddedDataset.set(i, j, 0.0);
    }
  }
}

size_t OperationMultipleEvalSubspaceAutoTuneTMP::getPaddedDatasetSize() {
  return paddedDatasetSize;
}

size_t OperationMultipleEvalSubspaceAutoTuneTMP::getAlignment() {
  return SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS;
}

std::string OperationMultipleEvalSubspaceAutoTuneTMP::getImplementationName() {
  return "SUBSPACEAUTOTUNETMP";
}

void OperationMultipleEvalSubspaceAutoTuneTMP::multTranspose(sgpp::base::DataVector &alpha,
                                                             sgpp::base::DataVector &result) {
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

  this->setCoefficients(result);

  if (!autotune::KernelMultTransposeSubspace.is_compiled()) {
    std::string sgpp_base_include;
    std::string boost_include;
    std::string autotunetmp_include;
    std::string vc_include;
    detail::get_includes_from_env(sgpp_base_include, boost_include, autotunetmp_include,
                                  vc_include);
    autotune::KernelMultTransposeSubspace.set_verbose(true);
    auto &builder = autotune::KernelMultTransposeSubspace.get_builder<cppjit::builder::gcc>();
    builder.set_include_paths(sgpp_base_include + boost_include + autotunetmp_include + vc_include);

    builder.set_cpp_flags(
        "-Wall -Wextra -Wno-unused-parameter -std=c++17 -march=native -mtune=native "
        "-O3 -g -ffast-math -fopenmp -fPIC -fno-gnu-unique -fopenmp");
    builder.set_link_flags("-shared -fno-gnu-unique -fopenmp");
    // autotune::KernelMultTransposeSubspace.set_source_dir("AutoTuneTMP_kernels/");

    // autotune::countable_set parameters;
    // autotune::fixed_set_parameter<size_t> p1("DATA_BLOCKING", {data_blocking});
    // parameters.add_parameter(p1);

    // size_t openmp_threads = omp_get_max_threads();
    // autotune::fixed_set_parameter<size_t> p2("KERNEL_OMP_THREADS", {openmp_threads});
    // parameters.add_parameter(p2);

    // autotune::fixed_set_parameter<size_t> p3("DIMS", {dims});
    // parameters.add_parameter(p3);

    // autotune::fixed_set_parameter<size_t> p4("ENTRIES", {dataset.getNrows()});
    // parameters.add_parameter(p4);

    // compile beforehand so that compilation is not part of the measured duration
    // autotune::KernelMultTransposeSubspace.set_parameter_values(parameters);
    autotune::KernelMultTransposeSubspace.compile();
  }

#pragma omp parallel
  {
    size_t start;
    size_t end;
    PartitioningTool::getOpenMPPartitionSegment(start_index_data, end_index_data, &start, &end,
                                                this->getAlignment());
    autotune::KernelMultTransposeSubspace(maxGridPointsOnLevel, isModLinear, paddedDataset,
                                          paddedDatasetSize, allSubspaceNodes, alpha, result, start,
                                          end);
  }

  this->unflatten(result);

  alpha.resize(originalAlphaSize);
  this->duration = this->timer.stop();
}

void OperationMultipleEvalSubspaceAutoTuneTMP::mult(sgpp::base::DataVector &source,
                                                    sgpp::base::DataVector &result) {
  if (!this->isPrepared) {
    this->prepare();
  }

  size_t originalResultSize = result.getSize();
  result.resizeZero(this->getPaddedDatasetSize());

  const size_t start_index_data = 0;
  const size_t end_index_data = this->getPaddedDatasetSize();

  this->timer.start();
  result.setAll(0.0);

  this->setCoefficients(source);

  if (!autotune::KernelMultSubspace.is_compiled()) {
    std::string sgpp_base_include;
    std::string boost_include;
    std::string autotunetmp_include;
    std::string vc_include;
    detail::get_includes_from_env(sgpp_base_include, boost_include, autotunetmp_include,
                                  vc_include);
    autotune::KernelMultSubspace.set_verbose(true);
    auto &builder = autotune::KernelMultSubspace.get_builder<cppjit::builder::gcc>();
    builder.set_include_paths(sgpp_base_include + boost_include + autotunetmp_include + vc_include);

    builder.set_cpp_flags(
        "-Wall -Wextra -Wno-unused-parameter -std=c++17 -march=native -mtune=native "
        "-O3 -g -ffast-math -fopenmp -fPIC -fno-gnu-unique -fopenmp");
    builder.set_link_flags("-shared -fno-gnu-unique -fopenmp");
    // autotune::KernelMultSubspace.set_source_dir("AutoTuneTMP_kernels/");

    // autotune::countable_set parameters;
    // autotune::fixed_set_parameter<size_t> p1("DATA_BLOCKING", {data_blocking});
    // parameters.add_parameter(p1);

    // size_t openmp_threads = omp_get_max_threads();
    // autotune::fixed_set_parameter<size_t> p2("KERNEL_OMP_THREADS", {openmp_threads});
    // parameters.add_parameter(p2);

    // autotune::fixed_set_parameter<size_t> p3("DIMS", {dims});
    // parameters.add_parameter(p3);

    // autotune::fixed_set_parameter<size_t> p4("ENTRIES", {dataset.getNrows()});
    // parameters.add_parameter(p4);

    // compile beforehand so that compilation is not part of the measured duration
    // autotune::KernelMultSubspace.set_parameter_values(parameters);
    autotune::KernelMultSubspace.compile();
  }

#pragma omp parallel
  {
    size_t start;
    size_t end;
    PartitioningTool::getOpenMPPartitionSegment(start_index_data, end_index_data, &start, &end,
                                                this->getAlignment());
    autotune::KernelMultSubspace(maxGridPointsOnLevel, isModLinear, paddedDataset,
                                 paddedDatasetSize, allSubspaceNodes, source, result, start, end);
  }

  // this->unflatten(result);

  result.resize(originalResultSize);

  this->duration = this->timer.stop();
}

}  // namespace sgpp::datadriven::SubspaceAutoTuneTMP
