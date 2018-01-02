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

#include "autotune/autotune.hpp"

AUTOTUNE_DECLARE_DEFINE_KERNEL(void(size_t, std::vector<double>, size_t, std::vector<double>,
                                    std::vector<double>, sgpp::base::DataVector&,
                                    std::vector<double>&),
                               streaming_mult_kernel)

// #include <opttmp/vectorization/vector_tiling.hpp>
#include <opttmp/vectorization/register_tiling.hpp>
using namespace opttmp::vectorization;  // dangerous!

namespace sgpp {
namespace datadriven {

// template <typename left, typename right>
// void compare_arrays(const left& l, const right& r, const std::string& where) {
//   for (size_t i = 0; i < l.size(); i++) {
//     for (size_t j = 0; j < double_v::size(); j++) {
//       double l_scalar = l[i][j];
//       double r_scalar = r[i][j];
//       // std::cout << "l_scalar: " << l_scalar << std::endl;
//       // std::cout << "r_scalar: " << r_scalar << std::endl;
//       if (l_scalar != r_scalar) {
//         // if (l_scalar != r[i][j]) {
//         std::cout << where << ", error i=" << i << ",j=" << j << ", " << l[i] << " != " << r[i]
//                   << std::endl;
//         // break;
//         std::terminate();
//       }
//     }
//   }
// }

class OperationMultiEvalStreamingAutoTuneTMP : public base::OperationMultipleEval {
  static constexpr size_t data_blocking = 5;  // SKL 8 significantly slower, 6 optimal?

 protected:
  size_t dims;

  std::vector<double> level_list_SoA;  // padded
  std::vector<double> index_list_SoA;  // padded
  size_t grid_size;                    // padded

  std::vector<double> level_list;  // not padded
  std::vector<double> index_list;  // not padded

  std::vector<double> dataset_SoA;  // padded
  size_t dataset_size;              // padded

  std::vector<double> dataset_non_SoA;  // not padded

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

    std::cout << "double_v::size(): " << double_v::size() << std::endl;

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < dataset_size; i += data_blocking * double_v::size()) {
      opttmp::vectorization::register_array<double_v, data_blocking> result_temps_arr(0.0);

      //////////////////////////////////////////////////////
      // std::array<double_v, data_blocking> result_temps;
      // for (size_t k = 0; k < data_blocking; k++) {
      //   result_temps[k] = 0.0;  // scalar initialization
      // }
      //////////////////////////////////////////////////////
      for (size_t j = 0; j < alpha.size(); j++) {
        opttmp::vectorization::register_array<double_v, data_blocking> evalNds_arr(alpha[j]);
        //////////////////////////////////////////////////////
        // std::array<double_v, data_blocking> evalNds;
        // for (size_t k = 0; k < data_blocking; k++) {
        //   evalNds[k] = alpha[j];  // broadcast, every processes same grid point
        // }
        //////////////////////////////////////////////////////

        // compare_arrays(evalNds_arr, evalNds, "alpha init");  // TODO: remove

        for (size_t d = 0; d < dims; d++) {
          // non-SoA is faster (2 streams, instead of 2d streams)
          // 2^l * x - i (level_list_SoA stores 2^l, not l)
          double_v level_dim = level_list[j * dims + d];  // broadcasts
          double_v index_dim = index_list[j * dims + d];

          opttmp::vectorization::register_array<double_v, data_blocking> data_dims_arr(
              &dataset_SoA[d * dataset_size + i], Vc::flags::element_aligned);
          //////////////////////////////////////////////////////
          // std::array<double_v, data_blocking> data_dims;
          // for (size_t k = 0; k < data_blocking; k++) {  // additional integer work
          //   data_dims[k] = double_v(&dataset_SoA[d * dataset_size + i + (k * double_v::size())],
          //                           Vc::flags::element_aligned);
          // }
          //////////////////////////////////////////////////////
          // compare_arrays(data_dims_arr, data_dims, "data_dims");  // TODO: remove

          // std::cout << "level_dim: " << level_dim << std::endl;
          // std::cout << "index_dim: " << index_dim << std::endl;
          opttmp::vectorization::register_array<double_v, data_blocking> temps_arr;
          temps_arr = (data_dims_arr * level_dim) - index_dim;
          //////////////////////////////////////////////////////
          // std::array<double_v, data_blocking> temps;          // no op
          // for (size_t k = 0; k < data_blocking; k++) {        // unrolled, no op
          //   temps[k] = level_dim * data_dims[k] - index_dim;  // 2 FLOPS (1 FMA)
          // }
          // compare_arrays(temps_arr, temps, "temps");              // TODO: remove
          // std::array<double_v, data_blocking> eval1ds;            // no op
          // for (size_t k = 0; k < data_blocking; k++) {            // unrolled, no op
          //   eval1ds[k] = Vc::max(one - Vc::abs(temps[k]), zero);  // 3 FLOPS
          // }
          //////////////////////////////////////////////////////
          opttmp::vectorization::register_array<double_v, data_blocking> eval1ds_arr;
          eval1ds_arr =
              opttmp::vectorization::max(-opttmp::vectorization::abs(temps_arr) + one, zero);
          // compare_arrays(eval1ds_arr, eval1ds, "eval1ds");  // TODO: remove

          //////////////////////////////////////////////////////
          // for (size_t k = 0; k < data_blocking; k++) {  // unrolled, no op
          //   evalNds[k] *= eval1ds[k];                   // 1 FLOPS
          // }
          //////////////////////////////////////////////////////
          evalNds_arr *= eval1ds_arr;
        }
        //////////////////////////////////////////////////////
        // for (size_t k = 0; k < data_blocking; k++) {  // unrolled, no op
        //   result_temps[k] += evalNds[k];              // total: 7d + 1 FLOPS
        // }
        //////////////////////////////////////////////////////
        result_temps_arr += evalNds_arr;
      }

      //////////////////////////////////////////////////////
      // for (size_t k = 0; k < data_blocking; k++) {  // additional integer work
      //   result_temps[k].memstore(&result_padded[i + (k * double_v::size())],
      //                            Vc::flags::element_aligned);
      // }
      //////////////////////////////////////////////////////
      result_temps_arr.memstore(&result_padded[i], Vc::flags::element_aligned);

      // compare_arrays(result_temps_arr, result_temps, "comparing result_temps*");
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
//           // 2^l * x - i (level_list_SoA stores 2^l, not l)
//           double_v level_dim = double_v(&level_list_SoA[d * grid_size + j],
//           Vc::flags::element_aligned);
//           double_v index_dim = double_v(&index_list_SoA[d * grid_size + j],
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

#pragma omp parallel for schedule(static)
    for (size_t j = 0; j < grid_size; j += double_v::size()) {
      double_v result_temp = 0.0;
      for (size_t i = 0; i < source.size(); i++) {
        double_v evalNd = source[i];

        for (size_t d = 0; d < dims; d++) {
          // 2^l * x - i (level_list_SoA stores 2^l, not l)
          double_v level_dim =
              double_v(&level_list_SoA[d * grid_size + j], Vc::flags::element_aligned);
          double_v index_dim =
              double_v(&index_list_SoA[d * grid_size + j], Vc::flags::element_aligned);

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
    level_list_SoA.resize(grid_size * dims);
    index_list_SoA.resize(grid_size * dims);
    for (size_t i = 0; i < storage.getSize(); i++) {
      base::HashGridPoint& point = storage[i];
      for (size_t d = 0; d < dims; d++) {
        base::GridPoint::level_type level;
        base::GridPoint::index_type index;
        point.get(d, level, index);
        level_list_SoA[d * grid_size + i] = static_cast<double>(1 << level);
        index_list_SoA[d * grid_size + i] = static_cast<double>(index);
      }
    }
    // setup padding
    base::HashGridPoint& last_point = storage[storage.getSize() - 1];
    for (size_t i = storage.getSize(); i < grid_size; i++) {
      for (size_t d = 0; d < dims; d++) {
        base::GridPoint::level_type level;
        base::GridPoint::index_type index;
        last_point.get(d, level, index);
        level_list_SoA[d * grid_size + i] = static_cast<double>(1 << level);
        index_list_SoA[d * grid_size + i] = static_cast<double>(index);
      }
    }
    // grid as non-SoA for testing
    level_list.resize(storage.getSize() * dims);
    index_list.resize(storage.getSize() * dims);
    for (size_t i = 0; i < storage.getSize(); i++) {
      base::HashGridPoint& point = storage[i];
      for (size_t d = 0; d < dims; d++) {
        base::GridPoint::level_type level;
        base::GridPoint::index_type index;
        point.get(d, level, index);
        level_list[i * dims + d] = static_cast<double>(1 << level);
        index_list[i * dims + d] = static_cast<double>(index);
      }
    }

    // data as SoA
    dataset_size = dataset.getNrows();
    dataset_size += dataset_size % (data_blocking * double_v::size()) == 0
                        ? 0
                        : (data_blocking * double_v::size()) -
                              (dataset_size % (data_blocking * double_v::size()));
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

    // data as non-SoA for testing
    dataset_non_SoA.resize(dataset.getNrows() * dims);
    for (size_t i = 0; i < dataset.getNrows(); i++) {
      for (size_t d = 0; d < dims; d++) {
        dataset_non_SoA[i * dims + d] = dataset[i * dims + d];
      }
    }
  }

  double getDuration() override { return duration; }
};

}  // namespace datadriven
}  // namespace sgpp