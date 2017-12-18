// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#pragma once

#include <omp.h>

#include "sgpp/base/exception/operation_exception.hpp"
#include "sgpp/base/operation/hash/OperationMultipleEval.hpp"
#include "sgpp/globaldef.hpp"

#include <chrono>

namespace sgpp {
namespace datadriven {

class OperationMultiEvalStreamingAutoTuneTMP : public base::OperationMultipleEval {
 protected:
  size_t dims;

  std::vector<double> level_list;
  std::vector<double> index_list;
  size_t grid_size;

  std::vector<double> dataset_SoA;
  size_t dataset_size;

  double duration;

 public:
  OperationMultiEvalStreamingAutoTuneTMP(base::Grid& grid, base::DataMatrix& dataset)
      : OperationMultipleEval(grid, dataset), dims(grid.getDimension()) {
    this->prepare();
  }

  ~OperationMultiEvalStreamingAutoTuneTMP() {}

  void mult(sgpp::base::DataVector& alpha, sgpp::base::DataVector& result) override {
    auto start = std::chrono::high_resolution_clock::now();

    // std::cout << "dims: " << dims << std::endl;
    // std::cout << "dataset_size: " << dataset_size << std::endl;
    // std::cout << "grid_size: " << grid_size << std::endl;

    // #pragma omp parallel for
    for (size_t i = 0; i < dataset_size; i++) {
      result[i] = 0.0;
      for (size_t j = 0; j < grid_size; j++) {
        double evalNd = alpha[j];

        for (size_t d = 0; d < dims; d++) {
          // 2^l * x - i (level_list stores 2^l, not l)
          double temp = level_list[d * grid_size + j] * dataset_SoA[d * dataset_size + i] -
                        index_list[d * grid_size + j];
          double eval1d = std::max<double>(1.0 - std::fabs(temp), 0.0);
          evalNd *= eval1d;
        }
        result[i] += evalNd;
      }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    duration = diff.count();
  }

  void multTranspose(sgpp::base::DataVector& source, sgpp::base::DataVector& result) override {
    auto start = std::chrono::high_resolution_clock::now();
    // #pragma omp parallel for
    for (size_t i = 0; i < dataset_size; i++) {
      for (size_t j = 0; j < grid_size; j++) {
        double evalNd = source[i];

        for (size_t d = 0; d < dims; d++) {
          double temp = level_list[d * grid_size + j] * dataset_SoA[d * dataset_size + i] -
                        index_list[d * grid_size + j];
          double eval1d = std::max<double>(1.0 - fabs(temp), 0.0);
          evalNd *= eval1d;
        }

        // #pragma omp atomic
        result[j] += evalNd;
      }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    duration = diff.count();
  }

  void prepare() override {
    // grid as SoA
    sgpp::base::GridStorage& storage = grid.getStorage();
    grid_size = storage.getSize();
    level_list.resize(grid_size * dims);
    index_list.resize(grid_size * dims);
    for (size_t i = 0; i < storage.getSize(); i++) {
      base::HashGridPoint& point = storage[i];
      for (size_t d = 0; d < dims; d++) {
        base::GridPoint::level_type level;
        base::GridPoint::index_type index;
        point.get(d, level, index);
        level_list[d * grid_size + i] = static_cast<double>(1 << level);
        index_list[d * grid_size + i] = static_cast<double>(index);
      }
    }
    // data as SoA
    dataset_size = dataset.getNrows();
    dataset_SoA.resize(dataset_size * dims);
    for (size_t i = 0; i < dataset_size; i++) {
      for (size_t d = 0; d < dims; d++) {
        dataset_SoA[d * dataset_size + i] = dataset[i * dims + d];
      }
    }
  }

  double getDuration() override { return duration; }
};

}  // namespace datadriven
}  // namespace sgpp
