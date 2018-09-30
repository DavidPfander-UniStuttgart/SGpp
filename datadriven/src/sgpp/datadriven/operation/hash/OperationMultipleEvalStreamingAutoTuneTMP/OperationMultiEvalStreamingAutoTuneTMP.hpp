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

#include <omp.h>

#include "autotune/autotune.hpp"
#include "autotune/parameter.hpp"
#include "autotune/tuners/bruteforce.hpp"
#include "autotune/tuners/countable_set.hpp"
#include "autotune/tuners/line_search.hpp"
#include "autotune/tuners/monte_carlo.hpp"
#include "autotune/tuners/neighborhood_search.hpp"
#include "autotune/tuners/randomizable_set.hpp"

#include <iomanip>

// AUTOTUNE_DECLARE_DEFINE_KERNEL(void(size_t, std::vector<double>&, size_t, std::vector<double>&,
//                                     std::vector<double>&, std::vector<double>&,
//                                     std::vector<double>&),
//                                streaming_mult_kernel)

AUTOTUNE_KERNEL(sgpp::base::DataVector(sgpp::base::Grid&, sgpp::base::DataMatrix&,
                                       sgpp::base::DataVector&, double&),
                streaming_mult_kernel, "AutoTuneTMP_kernels")

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
  static constexpr size_t data_blocking = 6;  // SKL 8 significantly slower, 6 optimal?

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
    // prepare is done by kernel
    // this->prepare();

    if (!autotune::streaming_mult_kernel.is_compiled()) {
      std::string sgpp_base_include;
      std::string boost_include;
      std::string autotunetmp_include;
      std::string vc_include;
      get_includes_from_env(sgpp_base_include, boost_include, autotunetmp_include, vc_include);
      autotune::streaming_mult_kernel.set_verbose(true);
      auto& builder = autotune::streaming_mult_kernel.get_builder<cppjit::builder::gcc>();
      builder.set_include_paths(sgpp_base_include + boost_include + autotunetmp_include +
                                vc_include);

      builder.set_cpp_flags(
          "-Wall -Wextra -Wno-unused-parameter -std=c++17 -march=native -mtune=native "
          "-O3 -g -ffast-math -fopenmp -fPIC -fno-gnu-unique -fopenmp");
      builder.set_link_flags("-shared -fno-gnu-unique -fopenmp");
      autotune::streaming_mult_kernel.set_source_dir("AutoTuneTMP_kernels/");

      autotune::countable_set parameters;
      autotune::fixed_set_parameter<size_t> p1("DATA_BLOCKING", {data_blocking});
      parameters.add_parameter(p1);

      size_t openmp_threads = omp_get_max_threads();
      autotune::fixed_set_parameter<size_t> p2("KERNEL_OMP_THREADS", {openmp_threads});
      parameters.add_parameter(p2);

      autotune::fixed_set_parameter<size_t> p3("DIMS", {dims});
      parameters.add_parameter(p3);

      autotune::fixed_set_parameter<size_t> p4("ENTRIES", {dataset.getNrows()});
      parameters.add_parameter(p4);

      // compile beforehand so that compilation is not part of the measured duration
      autotune::streaming_mult_kernel.set_parameter_values(parameters);
      autotune::streaming_mult_kernel.compile();
    }

    // auto start = std::chrono::high_resolution_clock::now();
    result = autotune::streaming_mult_kernel(grid, dataset, alpha, duration);
    // auto end = std::chrono::high_resolution_clock::now();

    // std::chrono::duration<double> diff = end - start;
    // duration = diff.count();
    double total_flops = dataset_size * alpha.size() * (6 * dims + 1);
    std::cout << "flop: " << total_flops << std::endl;
    std::cout << "gflops: " << ((total_flops * 1E-9) / duration) << std::endl;
  }

  void tune_mult(sgpp::base::DataVector& alpha, sgpp::base::DataVector& result,
                 const std::string& scenario_name, const std::string& tuner_name) {
    this->prepare();

    autotune::streaming_mult_kernel.set_verbose(true);

    auto& builder = autotune::streaming_mult_kernel.get_builder<cppjit::builder::gcc>();

    std::string sgpp_base_include;
    std::string boost_include;
    std::string autotunetmp_include;
    std::string vc_include;
    get_includes_from_env(sgpp_base_include, boost_include, autotunetmp_include, vc_include);
    builder.set_include_paths(sgpp_base_include + boost_include + autotunetmp_include + vc_include);
    builder.set_cpp_flags(
        "-Wall -Wextra -Wno-unused-parameter -std=c++17 -march=native -mtune=native "
        "-O3 -g -ffast-math -fopenmp -fPIC -fopenmp -fno-gnu-unique");
    builder.set_link_flags("-shared -fopenmp -fno-gnu-unique");

    autotune::streaming_mult_kernel.set_source_dir("AutoTuneTMP_kernels/");

    autotune::countable_set parameters;
    // autotune::fixed_set_parameter<size_t> p1("DATA_BLOCKING", {5, 6, 7, 8});
    autotune::countable_continuous_parameter p1("DATA_BLOCKING", 5.0, 1.0, 1.0, 15.0);
    parameters.add_parameter(p1);

    size_t openmp_threads = omp_get_max_threads();
    std::vector<size_t> thread_values;
    thread_values.push_back(openmp_threads);
    for (size_t i = 0; i < 1; i++) {  // 4-way HT assumed max
                                      // for (size_t i = 0; i < 3; i++) {  // 4-way HT assumed max
      if (openmp_threads % 2 == 0) {
        openmp_threads /= 2;
        thread_values.push_back(openmp_threads);
      } else {
        break;
      }
    }
    // TODO: handle NUMA with this parameter, too?
    std::cout << "KERNEL_OMP_THREADS: ";
    for (size_t i = 0; i < thread_values.size(); i++) {
      if (i > 0) {
        std::cout << ", ";
      }
      std::cout << thread_values[i];
    }
    std::cout << std::endl;
    autotune::fixed_set_parameter<size_t> p2("KERNEL_OMP_THREADS", thread_values);
    parameters.add_parameter(p2);

    autotune::fixed_set_parameter<size_t> p3("DIMS", {dims});
    parameters.add_parameter(p3);

    autotune::fixed_set_parameter<size_t> p4("ENTRIES", {dataset.getNrows()});
    parameters.add_parameter(p4);

    for (size_t i = 0; i < parameters.size(); i += 1) {
      parameters[i]->set_random_value();
    }

    autotune::randomizable_set parameters_randomizable;
    parameters_randomizable.add_parameter(p1);
    parameters_randomizable.add_parameter(p2);
    parameters_randomizable.add_parameter(p3);
    parameters_randomizable.add_parameter(p4);

    for (size_t i = 0; i < parameters_randomizable.size(); i += 1) {
      parameters_randomizable[i]->set_random_value();
    }

    sgpp::base::DataVector result_compare(dataset.getNrows());
    {
      result_compare.setAll(0.0);

      std::unique_ptr<sgpp::base::OperationMultipleEval> eval_compare =
          std::unique_ptr<sgpp::base::OperationMultipleEval>(
              sgpp::op_factory::createOperationMultipleEval(grid, dataset));

      eval_compare->mult(alpha, result_compare);
    }

    std::function<bool(sgpp::base::DataVector)> test_result =
        [result_compare](sgpp::base::DataVector result) -> bool {
      /* tests values generated by kernel */
      for (size_t i = 0; i < result_compare.size(); i++) {
        double diff = std::abs(result_compare[i] - result[i]) /
                      std::min(std::abs(result_compare[i]), std::abs(result[i]));
        if (diff >= 100 * std::numeric_limits<double>::epsilon()) {
          return false;
        }
      }
      return true;
    };

    double duration_compute;

    autotune::streaming_mult_kernel.set_kernel_duration_functor(
        [&duration_compute]() { return duration_compute; });

    autotune::countable_set optimal_parameters;

    if (tuner_name.compare("bruteforce") == 0) {
      autotune::tuners::bruteforce tuner(autotune::streaming_mult_kernel, parameters);
      tuner.set_verbose(true);
      tuner.set_write_measurement(scenario_name);
      tuner.setup_test(test_result);
      optimal_parameters = tuner.tune(grid, dataset, alpha, duration_compute);
    } else if (tuner_name.compare("line_search") == 0) {
      autotune::tuners::line_search tuner(autotune::streaming_mult_kernel, parameters,
                                          10 * parameters.size());
      tuner.set_verbose(true);
      tuner.set_write_measurement(scenario_name);
      tuner.setup_test(test_result);
      optimal_parameters = tuner.tune(grid, dataset, alpha, duration_compute);
    } else if (tuner_name.compare("neighborhood_search") == 0) {
      autotune::tuners::neighborhood_search tuner(autotune::streaming_mult_kernel, parameters, 100);
      tuner.set_verbose(true);
      tuner.set_write_measurement(scenario_name);
      tuner.setup_test(test_result);
      optimal_parameters = tuner.tune(grid, dataset, alpha, duration_compute);
    } else if (tuner_name.compare("monte_carlo") == 0) {
      autotune::tuners::monte_carlo tuner(autotune::streaming_mult_kernel, parameters_randomizable,
                                          100);
      tuner.set_verbose(true);
      tuner.set_write_measurement(scenario_name);
      tuner.setup_test(test_result);
      autotune::randomizable_set optimal_parameters_randomizable =
          tuner.tune(grid, dataset, alpha, duration_compute);
      autotune::streaming_mult_kernel.set_parameter_values(optimal_parameters_randomizable);
      autotune::streaming_mult_kernel.create_parameter_file();
      autotune::streaming_mult_kernel.compile();
    } else {
      throw "error: tuner not implemented!";
    }

    if (tuner_name.compare("monte_carlo") != 0) {
      // setup optimal parameters for next call to mult
      autotune::streaming_mult_kernel.set_parameter_values(optimal_parameters);
      autotune::streaming_mult_kernel.create_parameter_file();
      autotune::streaming_mult_kernel.compile();
    }
  }

  void multTranspose(sgpp::base::DataVector& source, sgpp::base::DataVector& result) override {
    throw "not implemented";
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

  void get_includes_from_env(std::string& sgpp_base_include, std::string& boost_include,
                             std::string& autotunetmp_include, std::string& vc_include) {
    const char* sgpp_base_include_env = std::getenv("SGPP_BASE_INCLUDE_DIR");
    if (sgpp_base_include_env) {
      sgpp_base_include = std::string("-I") + std::string(sgpp_base_include_env) + std::string(" ");
    } else {
      throw;
    }
    const char* boost_include_env = std::getenv("BOOST_INCLUDE_DIR");
    if (boost_include_env) {
      boost_include = std::string("-I") + std::string(boost_include_env) + std::string(" ");
    } else {
      throw;
    }
    const char* autotunetmp_include_env = std::getenv("AUTOTUNETMP_INCLUDE_DIR");
    if (autotunetmp_include_env) {
      autotunetmp_include =
          std::string("-I") + std::string(autotunetmp_include_env) + std::string(" ");
    } else {
      throw;
    }
    const char* vc_include_env = std::getenv("VC_INCLUDE_DIR");
    if (vc_include_env) {
      vc_include = std::string("-I") + std::string(vc_include_env) + std::string(" ");
    } else {
      throw;
    }
  }
};

}  // namespace datadriven
}  // namespace sgpp
