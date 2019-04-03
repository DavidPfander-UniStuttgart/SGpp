// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#include "OperationMultipleEvalSubspaceAutoTuneTMP.hpp"
#include "autotune/autotune.hpp"
#include "autotune/fixed_set_parameter.hpp"
#include "autotune/tuners/bruteforce.hpp"
#include "autotune/tuners/countable_set.hpp"
#include "autotune/tuners/line_search.hpp"
#include "autotune/tuners/monte_carlo.hpp"
#include "autotune/tuners/neighborhood_search.hpp"
#include "autotune/tuners/randomizable_set.hpp"
// #include <sgpp/datadriven/DatadrivenOpFactory.hpp>
#include "sgpp/base/operation/BaseOpFactory.hpp"
#include <fstream>
#include <string>
#include <vector>

using sgpp::base::DataMatrix;
using sgpp::base::DataVector;
using sgpp::base::Grid;

AUTOTUNE_KERNEL(
    void(size_t, bool, sgpp::base::DataMatrix &, size_t,
         std::vector<sgpp::datadriven::SubspaceAutoTuneTMP::SubspaceNode> &,
         sgpp::base::DataVector &, sgpp::base::DataVector &, size_t, size_t),
    KernelMultSubspace, "SubspaceAutoTuneTMPKernels/Mult")

AUTOTUNE_KERNEL(
    void(size_t, bool, sgpp::base::DataMatrix &, size_t,
         std::vector<sgpp::datadriven::SubspaceAutoTuneTMP::SubspaceNode> &,
         sgpp::base::DataVector &, sgpp::base::DataVector &, size_t, size_t),
    KernelMultTransposeSubspace, "SubspaceAutoTuneTMPKernels/MultTranspose")

namespace sgpp::datadriven::SubspaceAutoTuneTMP {

namespace detail {
void get_includes_from_env(std::string &sgpp_base_include,
                           std::string &boost_include,
                           std::string &autotunetmp_include,
                           std::string &vc_include) {
  const char *sgpp_base_include_env = std::getenv("SGPP_BASE_INCLUDE_DIR");
  if (sgpp_base_include_env) {
    sgpp_base_include = std::string("-I") + std::string(sgpp_base_include_env) +
                        std::string(" ");
  } else {
    throw;
  }
  const char *boost_include_env = std::getenv("BOOST_INCLUDE_DIR");
  if (boost_include_env) {
    boost_include =
        std::string("-I") + std::string(boost_include_env) + std::string(" ");
  } else {
    throw;
  }
  const char *autotunetmp_include_env = std::getenv("AUTOTUNETMP_INCLUDE_DIR");
  if (autotunetmp_include_env) {
    autotunetmp_include = std::string("-I") +
                          std::string(autotunetmp_include_env) +
                          std::string(" ");
  } else {
    throw;
  }
  const char *vc_include_env = std::getenv("VC_INCLUDE_DIR");
  if (vc_include_env) {
    vc_include =
        std::string("-I") + std::string(vc_include_env) + std::string(" ");
  } else {
    throw;
  }
}
} // namespace detail

OperationMultipleEvalSubspaceAutoTuneTMP::
    OperationMultipleEvalSubspaceAutoTuneTMP(Grid &grid, DataMatrix &dataset,
                                             bool isModLinear)
    : base::OperationMultipleEval(grid, dataset), storage(grid.getStorage()),
      duration(-1.0), paddedDatasetSize(0), maxGridPointsOnLevel(0),
      dim(dataset.getNcols()), maxLevel(0), subspaceCount(-1),
      totalRegularGridPoints(-1), isModLinear(isModLinear), refinementStep(0),
      csvSep(","), write_stats(false), listRatio(0.2), streamingThreshold(128),
      parallelDataPoints(256), vectorPadding(4) {
  this->padDataset(dataset);
}

OperationMultipleEvalSubspaceAutoTuneTMP::
    ~OperationMultipleEvalSubspaceAutoTuneTMP() {}

void OperationMultipleEvalSubspaceAutoTuneTMP::prepare() {
  this->prepareSubspaceIterator();
  this->isPrepared = true;
}

void OperationMultipleEvalSubspaceAutoTuneTMP::setCoefficients(
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

void OperationMultipleEvalSubspaceAutoTuneTMP::setSurplus(
    std::vector<uint32_t> &level, std::vector<uint32_t> &maxIndices,
    std::vector<uint32_t> &index, double value) {
  uint32_t levelFlat = this->flattenLevel(this->dim, this->maxLevel, level);
  uint32_t indexFlat = this->flattenIndex(this->dim, maxIndices, index);
  uint32_t subspaceIndex = this->allLevelsIndexMap.find(levelFlat)->second;
  SubspaceNode &subspace = this->allSubspaceNodes[subspaceIndex];
  subspace.setSurplus(indexFlat, value);
}

void OperationMultipleEvalSubspaceAutoTuneTMP::getSurplus(
    std::vector<uint32_t> &level, std::vector<uint32_t> &maxIndices,
    std::vector<uint32_t> &index, double &value, bool &isVirtual) {
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

uint32_t OperationMultipleEvalSubspaceAutoTuneTMP::flattenLevel(
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

void OperationMultipleEvalSubspaceAutoTuneTMP::padDataset(
    sgpp::base::DataMatrix &dataset) {
  size_t chunkSize = parallelDataPoints;

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

  paddedDataset.resize(paddedDataset.size() +
                       vectorPadding * 2 * paddedDataset.getNcols());

  for (size_t i = paddedDatasetSize; i < paddedDataset.getNrows(); i += 1) {
    for (size_t j = 0; j < paddedDataset.getNcols(); j += 1) {
      paddedDataset.set(i, j, 0.0);
    }
  }
}

size_t OperationMultipleEvalSubspaceAutoTuneTMP::getPaddedDatasetSize() {
  return paddedDatasetSize;
}

std::string OperationMultipleEvalSubspaceAutoTuneTMP::getImplementationName() {
  return "SUBSPACEAUTOTUNETMP";
}

void OperationMultipleEvalSubspaceAutoTuneTMP::multTranspose(
    sgpp::base::DataVector &source, sgpp::base::DataVector &result) {
  if (!this->isPrepared) {
    this->prepare();
  }

  size_t originalSourceSize = source.getSize();

  const size_t start_index_data = 0;
  const size_t end_index_data = this->getPaddedDatasetSize();

  // pad the source vector to the padded size of the dataset
  source.resizeZero(this->getPaddedDatasetSize());

  this->timer.start();
  result.setAll(0.0);

  this->setCoefficients(result);

  if (!autotune::KernelMultTransposeSubspace.is_compiled()) {
    std::string sgpp_base_include;
    std::string boost_include;
    std::string autotunetmp_include;
    std::string vc_include;
    detail::get_includes_from_env(sgpp_base_include, boost_include,
                                  autotunetmp_include, vc_include);
    autotune::KernelMultTransposeSubspace.set_verbose(true);
    auto &builder = autotune::KernelMultTransposeSubspace
                        .get_builder<cppjit::builder::gcc>();
    builder.set_include_paths(sgpp_base_include + boost_include +
                              autotunetmp_include + vc_include);

    // warning: this kernel is not compatibel with "-ffast-math"
    builder.set_cpp_flags(
        "-Wall -Wextra -Wno-unused-parameter -std=c++17 -march=native "
        "-mtune=native "
        "-O3 -g -fopenmp -fPIC -fno-gnu-unique -fopenmp");
    builder.set_link_flags("-shared -fno-gnu-unique -fopenmp");
    builder.set_do_cleanup(false);

    autotune::countable_set parameters;
    autotune::fixed_set_parameter<int64_t> p0(
        "SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS", {256});
    autotune::fixed_set_parameter<int64_t> p1(
        "SUBSPACEAUTOTUNETMP_ENABLE_SUBSPACE_SKIPPING", {1});
    autotune::fixed_set_parameter<int64_t> p2("SUBSPACEAUTOTUNETMP_UNROLL",
                                              {0});
    autotune::fixed_set_parameter<int64_t> p3("SUBSPACEAUTOTUNETMP_VEC_PADDING",
                                              std::vector<int64_t>{4});
    autotune::fixed_set_parameter<int64_t> p4(
        "SUBSPACEAUTOTUNETMP_STREAMING_THRESHOLD", std::vector<int64_t>{128});
    autotune::fixed_set_parameter<double> p5("SUBSPACEAUTOTUNETMP_LIST_RATIO",
                                             std::vector<double>{0.2});
    autotune::fixed_set_parameter<int64_t> p6(
        "SUBSPACEAUTOTUNETMP_ENABLE_PARTIAL_RESULT_REUSAGE",
        std::vector<int64_t>{1});

    parameters.add_parameter(p0);
    parameters.add_parameter(p1);
    parameters.add_parameter(p2);
    parameters.add_parameter(p3);
    parameters.add_parameter(p4);
    parameters.add_parameter(p5);
    parameters.add_parameter(p6);

    autotune::KernelMultTransposeSubspace.set_parameter_values(parameters);
    autotune::KernelMultTransposeSubspace.compile();
  }

#pragma omp parallel
  {
    size_t start;
    size_t end;
    PartitioningTool::getOpenMPPartitionSegment(
        start_index_data, end_index_data, &start, &end, parallelDataPoints);
    autotune::KernelMultTransposeSubspace(
        maxGridPointsOnLevel, isModLinear, paddedDataset, paddedDatasetSize,
        allSubspaceNodes, source, result, start, end);
  }

  this->unflatten(result);
  source.resize(originalSourceSize);
  this->duration = this->timer.stop();
}

void OperationMultipleEvalSubspaceAutoTuneTMP::mult(
    sgpp::base::DataVector &alpha, sgpp::base::DataVector &result) {

  if (!autotune::KernelMultSubspace.is_compiled()) {
    std::string sgpp_base_include;
    std::string boost_include;
    std::string autotunetmp_include;
    std::string vc_include;
    detail::get_includes_from_env(sgpp_base_include, boost_include,
                                  autotunetmp_include, vc_include);
    autotune::KernelMultSubspace.set_verbose(true);
    auto &builder =
        autotune::KernelMultSubspace.get_builder<cppjit::builder::gcc>();
    builder.set_include_paths(sgpp_base_include + boost_include +
                              autotunetmp_include + vc_include);

    // warning: this kernel is not compatibel with "-ffast-math"
    builder.set_cpp_flags(
        "-Wall -Wextra -Wno-unused-parameter -std=c++17 -march=native "
        "-mtune=native "
        "-O3 -g -fopenmp -fPIC -fno-gnu-unique -fopenmp");
    builder.set_link_flags("-shared -fno-gnu-unique -fopenmp");
    builder.set_do_cleanup(false);

    autotune::countable_set parameters;
    autotune::fixed_set_parameter<int64_t> p0(
        "SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS", {256});
    autotune::fixed_set_parameter<int64_t> p1(
        "SUBSPACEAUTOTUNETMP_ENABLE_SUBSPACE_SKIPPING", {1});
    autotune::fixed_set_parameter<int64_t> p2("SUBSPACEAUTOTUNETMP_UNROLL",
                                              {0});
    autotune::fixed_set_parameter<int64_t> p3("SUBSPACEAUTOTUNETMP_VEC_PADDING",
                                              std::vector<int64_t>{4});
    autotune::fixed_set_parameter<int64_t> p4(
        "SUBSPACEAUTOTUNETMP_STREAMING_THRESHOLD", std::vector<int64_t>{128});
    autotune::fixed_set_parameter<double> p5("SUBSPACEAUTOTUNETMP_LIST_RATIO",
                                             std::vector<double>{0.2});
    autotune::fixed_set_parameter<int64_t> p6(
        "SUBSPACEAUTOTUNETMP_ENABLE_PARTIAL_RESULT_REUSAGE",
        std::vector<int64_t>{1});

    parameters.add_parameter(p0);
    parameters.add_parameter(p1);
    parameters.add_parameter(p2);
    parameters.add_parameter(p3);
    parameters.add_parameter(p4);
    parameters.add_parameter(p5);
    parameters.add_parameter(p6);

    autotune::KernelMultSubspace.set_parameter_values(parameters);
    autotune::KernelMultSubspace.compile();
  }

  if (!this->isPrepared) {
    this->prepare();
  }

  size_t originalResultSize = result.getSize();
  result.resizeZero(this->getPaddedDatasetSize());

  const size_t start_index_data = 0;
  const size_t end_index_data = this->getPaddedDatasetSize();

  this->timer.start();
  result.setAll(0.0);

  this->setCoefficients(alpha);

#pragma omp parallel
  {
    size_t start;
    size_t end;
    PartitioningTool::getOpenMPPartitionSegment(
        start_index_data, end_index_data, &start, &end, parallelDataPoints);
    autotune::KernelMultSubspace(maxGridPointsOnLevel, isModLinear,
                                 paddedDataset, paddedDatasetSize,
                                 allSubspaceNodes, alpha, result, start, end);
  }

  result.resize(originalResultSize);
  this->duration = this->timer.stop();
}

void OperationMultipleEvalSubspaceAutoTuneTMP::set_write_stats(
    const std::string &stats_file_name, const std::string &name) {
  if (statsFile.is_open()) {
    statsFile.close();
  }
  statsFile.open(stats_file_name);
  statsFile << "# name: " << name << std::endl;
  statsFile << "refinementStep & ";
  statsFile << "nonVirtualGridPoints & ";
  statsFile << "totalRegularGridPoints & ";
  statsFile << "actualGridPoints & ";
  statsFile << "largestArraySubspace & ";
  statsFile << "largestListSubspace & ";
  statsFile << "numberOfListSubspaces & ";
  statsFile << "subspaceCount & ";
  statsFile << "avrPointsPerSubspace & ";
  statsFile << "memoryEstimate & ";
  statsFile << "memoryEfficiency";
  statsFile << std::endl;
  write_stats = true;
}

double OperationMultipleEvalSubspaceAutoTuneTMP::getDuration() {
  return this->duration;
}

// void OperationMultipleEvalSubspaceAutoTuneTMP::tune_mult(
//     sgpp::base::DataVector &alpha, sgpp::base::DataVector &result,
//     const std::string &scenario_name, const std::string &tuner_name) {
//   this->prepare();

//   autotune::KernelMultSubspace.set_verbose(true);

//   auto &builder =
//       autotune::KernelMultSubspace.get_builder<cppjit::builder::gcc>();

//   std::string sgpp_base_include;
//   std::string boost_include;
//   std::string autotunetmp_include;
//   std::string vc_include;
//   detail::get_includes_from_env(sgpp_base_include, boost_include,
//                                 autotunetmp_include, vc_include);
//   builder.set_include_paths(sgpp_base_include + boost_include +
//                             autotunetmp_include + vc_include);
//   builder.set_cpp_flags(
//       "-Wall -Wextra -Wno-unused-parameter -std=c++17 -march=native "
//       "-mtune=native "
//       "-O3 -g -ffast-math -fopenmp -fPIC -fopenmp -fno-gnu-unique");
//   builder.set_link_flags("-shared -fopenmp -fno-gnu-unique");

//   autotune::KernelMultSubspace.set_source_dir("AutoTuneTMP_kernels/");

//   autotune::countable_set parameters;
//   autotune::fixed_set_parameter<int64_t> p0(
//       "SUBSPACEAUTOTUNETMP_PARALLEL_DATA_POINTS", {256});
//   autotune::fixed_set_parameter<int64_t> p1(
//       "SUBSPACEAUTOTUNETMP_ENABLE_SUBSPACE_SKIPPING", {1});
//   autotune::fixed_set_parameter<int64_t> p2("SUBSPACEAUTOTUNETMP_UNROLL", {0});
//   autotune::fixed_set_parameter<int64_t> p3("SUBSPACEAUTOTUNETMP_VEC_PADDING",
//                                             std::vector<int64_t>{4});
//   autotune::fixed_set_parameter<int64_t> p4(
//       "SUBSPACEAUTOTUNETMP_STREAMING_THRESHOLD", std::vector<int64_t>{128});
//   autotune::fixed_set_parameter<double> p5("SUBSPACEAUTOTUNETMP_LIST_RATIO",
//                                            std::vector<double>{0.2});
//   autotune::fixed_set_parameter<int64_t> p6(
//       "SUBSPACEAUTOTUNETMP_ENABLE_PARTIAL_RESULT_REUSAGE",
//       std::vector<int64_t>{1});

//   parameters.add_parameter(p0);
//   parameters.add_parameter(p1);
//   parameters.add_parameter(p2);
//   parameters.add_parameter(p3);
//   parameters.add_parameter(p4);
//   parameters.add_parameter(p5);
//   parameters.add_parameter(p6);

//   autotune::randomizable_set parameters_randomizable;
//   parameters_randomizable.add_parameter(p0);
//   parameters_randomizable.add_parameter(p1);
//   parameters_randomizable.add_parameter(p2);
//   parameters_randomizable.add_parameter(p3);
//   parameters_randomizable.add_parameter(p4);
//   parameters_randomizable.add_parameter(p5);
//   parameters_randomizable.add_parameter(p6);

//   for (size_t i = 0; i < parameters_randomizable.size(); i += 1) {
//     parameters_randomizable[i]->set_random_value();
//   }

//   sgpp::base::DataVector result_compare(dataset.getNrows());
//   {
//     result_compare.setAll(0.0);

//     std::unique_ptr<sgpp::base::OperationMultipleEval> eval_compare =
//         std::unique_ptr<sgpp::base::OperationMultipleEval>(
//             sgpp::op_factory::createOperationMultipleEval(grid, dataset));

//     eval_compare->mult(alpha, result_compare);
//   }

//   // std::function<bool(sgpp::base::DataVector)> test_result =
//   //     [result_compare](sgpp::base::DataVector result) -> bool {
//   //   /* tests values generated by kernel */
//   //   for (size_t i = 0; i < result_compare.size(); i++) {
//   //     double diff = std::abs(result_compare[i] - result[i]) /
//   //                   std::min(std::abs(result_compare[i]),
//   //                   std::abs(result[i]));
//   //     if (diff >= 100 * std::numeric_limits<double>::epsilon()) {
//   //       return false;
//   //     }
//   //   }
//   //   return true;
//   // };

//   double duration_compute;

//   autotune::KernelMultSubspace.set_kernel_duration_functor(
//       [this]() { return this->duration; });

//   autotune::countable_set optimal_parameters;

//   if (tuner_name.compare("bruteforce") == 0) {
//     autotune::tuners::bruteforce tuner(autotune::KernelMultSubspace,
//                                        parameters);
//     tuner.set_verbose(true);
//     tuner.set_write_measurement(scenario_name);
//     // tuner.setup_test(test_result);
//     optimal_parameters = tuner.tune(grid, dataset, alpha, duration_compute);
//   } else if (tuner_name.compare("line_search") == 0) {
//     autotune::tuners::line_search tuner(autotune::KernelMultSubspace,
//                                         parameters, 10 * parameters.size());
//     tuner.set_verbose(true);
//     tuner.set_write_measurement(scenario_name);
//     // tuner.setup_test(test_result);
//     optimal_parameters = tuner.tune(grid, dataset, alpha, duration_compute);
//   } else if (tuner_name.compare("neighborhood_search") == 0) {
//     autotune::tuners::neighborhood_search tuner(autotune::KernelMultSubspace,
//                                                 parameters, 100);
//     tuner.set_verbose(true);
//     tuner.set_write_measurement(scenario_name);
//     // tuner.setup_test(test_result);
//     optimal_parameters = tuner.tune(grid, dataset, alpha, duration_compute);
//   } else if (tuner_name.compare("monte_carlo") == 0) {
//     autotune::tuners::monte_carlo tuner(autotune::KernelMultSubspace,
//                                         parameters_randomizable, 100);
//     tuner.set_verbose(true);
//     tuner.set_write_measurement(scenario_name);
//     // tuner.setup_test(test_result);
//     autotune::randomizable_set optimal_parameters_randomizable =
//         tuner.tune(grid, dataset, alpha, duration_compute);
//     autotune::KernelMultSubspace.set_parameter_values(
//         optimal_parameters_randomizable);
//     autotune::KernelMultSubspace.create_parameter_file();
//     autotune::KernelMultSubspace.compile();
//   } else {
//     throw "error: tuner not implemented!";
//   }

//   if (tuner_name.compare("monte_carlo") != 0) {
//     // setup optimal parameters for next call to mult
//     autotune::KernelMultSubspace.set_parameter_values(optimal_parameters);
//     autotune::KernelMultSubspace.create_parameter_file();
//     autotune::KernelMultSubspace.compile();
//   }
// }

} // namespace sgpp::datadriven::SubspaceAutoTuneTMP
