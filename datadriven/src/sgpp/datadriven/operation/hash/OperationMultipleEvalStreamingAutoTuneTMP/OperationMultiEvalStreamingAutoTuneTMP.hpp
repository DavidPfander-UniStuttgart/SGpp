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
  static constexpr size_t data_blocking = 10;

 protected:
  size_t dims;

  std::vector<double> level_list;
  std::vector<double> index_list;
  size_t grid_size;

  std::vector<double> dataset_SoA;
  size_t dataset_size;

  double duration;
  bool verbose;

 public:
  OperationMultiEvalStreamingAutoTuneTMP(base::Grid& grid, base::DataMatrix& dataset, bool verbose)
      : OperationMultipleEval(grid, dataset), dims(grid.getDimension()), verbose(verbose) {
    this->prepare();
  }

  ~OperationMultiEvalStreamingAutoTuneTMP() {}

  void mult(sgpp::base::DataVector& alpha, sgpp::base::DataVector& result) override {
    this->prepare();

    std::vector<double> result_padded(dataset_size);

    auto start = std::chrono::high_resolution_clock::now();

    const double_v one = 1.0;
    const double_v zero = 0.0;

// #pragma omp parallel for
//     for (size_t i = 0; i < dataset_size; i += double_v::size()) {
//       double_v result_temp = 0.0;
//       for (size_t j = 0; j < alpha.size(); j++) {
//         double_v evalNd = alpha[j];

//         for (size_t d = 0; d < dims; d++) {
//           // TODO: non-SoA probably faster
//           // 2^l * x - i (level_list stores 2^l, not l)
//           double_v level_dim = level_list[d * grid_size + j];
//           double_v index_dim = index_list[d * grid_size + j];

//           double_v data_dim =
//               double_v(&dataset_SoA[d * dataset_size + i], Vc::flags::element_aligned);
//           double_v temp = level_dim * data_dim - index_dim;      // 2 FLOPS
//           double_v eval1d = Vc::max(one - Vc::abs(temp), zero);  // 3 FLOPS
//           evalNd *= eval1d;                                      // 1 FLOPS
//         }
//         result_temp += evalNd;  // total: 7d + 1 FLOPS
//       }
//       result_temp.memstore(&result_padded[i], Vc::flags::element_aligned);
//     }

#pragma omp parallel for
    for (size_t i = 0; i < dataset_size; i += data_blocking * double_v::size()) {
      // double_v result_temp = 0.0;
      std::array<double_v, data_blocking> result_temps;
      for (size_t k = 0; k < data_blocking; k++) {
        result_temps[k] = 0.0;
      }
      for (size_t j = 0; j < alpha.size(); j++) {
        // double_v evalNd = alpha[j];
        std::array<double_v, data_blocking> evalNds;
        for (size_t k = 0; k < data_blocking; k++) {
          evalNds[k] = alpha[j];
        }

        for (size_t d = 0; d < dims; d++) {
          // TODO: non-SoA probably faster
          // 2^l * x - i (level_list stores 2^l, not l)
          double_v level_dim = level_list[d * grid_size + j];
          double_v index_dim = index_list[d * grid_size + j];

          // double_v data_dim =
          //     double_v(&dataset_SoA[d * dataset_size + i], Vc::flags::element_aligned);
          std::array<double_v, data_blocking> data_dims;
          for (size_t k = 0; k < data_blocking; k++) {
            data_dims[k] = double_v(&dataset_SoA[d * dataset_size + i + (k * double_v::size())],
                                    Vc::flags::element_aligned);
          }
          // double_v temp = level_dim * data_dim - index_dim;      // 2 FLOPS
          // double_v eval1d = Vc::max(one - Vc::abs(temp), zero);  // 3 FLOPS
          // evalNd *= eval1d;                                      // 1 FLOPS

          for (size_t k = 0; k < data_blocking; k++) {
            double_v temp = level_dim * data_dims[k] - index_dim;  // 2 FLOPS
            double_v eval1d = Vc::max(one - Vc::abs(temp), zero);  // 3 FLOPS
            evalNds[k] *= eval1d;                                  // 1 FLOPS
          }
        }
        // result_temp += evalNd;  // total: 7d + 1 FLOPS
        for (size_t k = 0; k < data_blocking; k++) {
          result_temps[k] += evalNds[k];  // total: 7d + 1 FLOPS
        }
      }
      // result_temp.memstore(&result_padded[i], Vc::flags::element_aligned);
      for (size_t k = 0; k < data_blocking; k++) {
        result_temps[k].memstore(&result_padded[i + (k * double_v::size())],
                                 Vc::flags::element_aligned);
      }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    duration = diff.count();
    double total_flops = dataset_size * alpha.size() * (6 * dims + 1);
    std::cout << "flop: " << total_flops << std::endl;
    std::cout << "gflops: " << ((total_flops * 1E-9) / duration) << std::endl;

    for (size_t i = 0; i < result.size(); i++) {
      result[i] = result_padded[i];
    }
  }

  void multTranspose(sgpp::base::DataVector& source, sgpp::base::DataVector& result) override {
    this->prepare();

    std::vector<double> result_padded(grid_size);

    auto start = std::chrono::high_resolution_clock::now();

    const double_v one = 1.0;
    const double_v zero = 0.0;

// #pragma omp parallel for
//     for (size_t j = 0; j < grid_size; j += double_v::size()) {
//       double_v result_temp = 0.0;
//       for (size_t i = 0; i < source.size(); i++) {
//         double_v evalNd = source[i];

//         for (size_t d = 0; d < dims; d++) {
//           // 2^l * x - i (level_list stores 2^l, not l)
//           double_v level_dim = double_v(&level_list[d * grid_size + j],
//           Vc::flags::element_aligned);
//           double_v index_dim = double_v(&index_list[d * grid_size + j],
//           Vc::flags::element_aligned);
//           // TODO: non-SoA probably faster
//           double_v data_dim = dataset_SoA[d * dataset_size + i];
//           double_v temp = level_dim * data_dim - index_dim;      // 2 FLOPS
//           double_v eval1d = Vc::max(one - Vc::abs(temp), zero);  // 3 FLOPS
//           evalNd *= eval1d;                                      // 1 FLOPS
//         }
//         result_temp += evalNd;  // total: 6d + 1
//       }
//       // #pragma omp atomic
//       result_temp.memstore(&result_padded[j], Vc::flags::element_aligned);
//     }

#pragma omp parallel for
    for (size_t j = 0; j < grid_size; j += double_v::size()) {
      double_v result_temp = 0.0;
      for (size_t i = 0; i < source.size(); i++) {
        double_v evalNd = source[i];

        for (size_t d = 0; d < dims; d++) {
          // 2^l * x - i (level_list stores 2^l, not l)
          double_v level_dim = double_v(&level_list[d * grid_size + j], Vc::flags::element_aligned);
          double_v index_dim = double_v(&index_list[d * grid_size + j], Vc::flags::element_aligned);

          // TODO: non-SoA probably faster
          double_v data_dim = dataset_SoA[d * dataset_size + i];
          double_v temp = level_dim * data_dim - index_dim;      // 2 FLOPS
          double_v eval1d = Vc::max(one - Vc::abs(temp), zero);  // 3 FLOPS
          evalNd *= eval1d;                                      // 1 FLOPS
        }
        result_temp += evalNd;  // total: 6d + 1
      }
      // #pragma omp atomic
      result_temp.memstore(&result_padded[j], Vc::flags::element_aligned);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    duration = diff.count();

    double total_flops = grid_size * source.size() * (6 * dims + 1);
    std::cout << "flop: " << total_flops << std::endl;
    std::cout << "gflops: " << ((total_flops * 1E-9) / duration) << std::endl;

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
