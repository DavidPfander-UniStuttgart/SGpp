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

#include <Vc/Vc>
using Vc::double_v;

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
    this->prepare();
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<double> result_padded(dataset_size);

    const double_v one = 1.0;
    const double_v zero = 0.0;

#pragma omp parallel for
    for (size_t i = 0; i < dataset_size; i += double_v::size()) {
      double_v result_temp = 0.0;
      for (size_t j = 0; j < alpha.size(); j++) {
        double_v evalNd = alpha[j];

        for (size_t d = 0; d < dims; d++) {
          // 2^l * x - i (level_list stores 2^l, not l)
          double_v level_dim = level_list[d * grid_size + j];
          double_v index_dim = index_list[d * grid_size + j];
          double_v data_dim =
              double_v(&dataset_SoA[d * dataset_size + i], Vc::flags::element_aligned);
          double_v temp = level_dim * data_dim - index_dim;
          double_v eval1d = Vc::max(one - Vc::abs(temp), zero);
          evalNd *= eval1d;
        }
        result_temp += evalNd;
      }
      result_temp.memstore(&result_padded[i], Vc::flags::element_aligned);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    duration = diff.count();

    for (size_t i = 0; i < result.size(); i++) {
      result[i] = result_padded[i];
    }
  }

  void multTranspose(sgpp::base::DataVector& source, sgpp::base::DataVector& result) override {
    this->prepare();

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<double> result_padded(grid_size);

    const double_v one = 1.0;
    const double_v zero = 0.0;

#pragma omp parallel for
    for (size_t j = 0; j < grid_size; j += double_v::size()) {
      double_v result_temp = 0.0;
      for (size_t i = 0; i < source.size(); i++) {
        double_v evalNd = source[i];

        for (size_t d = 0; d < dims; d++) {
          // 2^l * x - i (level_list stores 2^l, not l)
          double_v level_dim = double_v(&level_list[d * grid_size + j], Vc::flags::element_aligned);
          double_v index_dim = double_v(&index_list[d * grid_size + j], Vc::flags::element_aligned);
          double_v data_dim = dataset_SoA[d * dataset_size + i];
          double_v temp = level_dim * data_dim - index_dim;
          double_v eval1d = Vc::max(one - Vc::abs(temp), zero);
          evalNd *= eval1d;
        }
        result_temp += evalNd;
      }
      // #pragma omp atomic
      result_temp.memstore(&result_padded[j], Vc::flags::element_aligned);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    duration = diff.count();

    for (size_t i = 0; i < result.size(); i++) {
      result[i] = result_padded[i];
    }
  }

  void prepare() override {
    // grid as SoA
    sgpp::base::GridStorage& storage = grid.getStorage();
    grid_size = storage.getSize();
    grid_size +=
        grid_size % double_v::size() == 0 ? 0 : double_v::size() - grid_size % double_v::size();
    std::cout << "double_v::size(): " << double_v::size() << std::endl;
    std::cout << "storage.size(): " << storage.getSize() << std::endl;
    std::cout << "grid_size: " << grid_size << std::endl;
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
    // setup padding
    base::HashGridPoint& last_point = storage[storage.getSize() - 1];
    for (size_t i = storage.getSize(); i < grid_size; i++) {
      for (size_t d = 0; d < dims; d++) {
        base::GridPoint::level_type level;
        base::GridPoint::index_type index;
        last_point.get(d, level, index);
        level_list[d * grid_size + i] = static_cast<double>(1 << level);
        index_list[d * grid_size + i] = static_cast<double>(index);
      }
    }

    // data as SoA
    dataset_size = dataset.getNrows();
    dataset_size += dataset_size % double_v::size() == 0 ? 0 : dataset_size % double_v::size();
    dataset_SoA.resize(dataset_size * dims);
    for (size_t i = 0; i < dataset.getNrows(); i++) {
      for (size_t d = 0; d < dims; d++) {
        dataset_SoA[d * dataset_size + i] = dataset[i * dims + d];
      }
    }
    // setup padding
    for (size_t i = dataset.getNrows(); i < dataset_size; i++) {
      for (size_t d = 0; d < dims; d++) {
        dataset_SoA[d * dataset_size + i] = dataset[(dataset.getNrows() - 1) * dims + d];
      }
    }
  }

  double getDuration() override { return duration; }
};

}  // namespace datadriven
}  // namespace sgpp
