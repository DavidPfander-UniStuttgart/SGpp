// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#pragma once

#include <algorithm>
#include <chrono>
#include <mutex> // NOLINT(build/c++11)
#include <sstream>
#include <vector>

#include "sgpp/base/exception/operation_exception.hpp"
#include "sgpp/base/opencl/OCLManagerMultiPlatform.hpp"
#include "sgpp/base/opencl/OCLOperationConfiguration.hpp"
#include "sgpp/base/operation/hash/OperationMultipleEval.hpp"
#include "sgpp/datadriven/DatadrivenOpFactory.hpp"
#include "sgpp/datadriven/operation/hash/DatadrivenOperationCommon.hpp"
#include "sgpp/datadriven/operation/hash/OperationMultipleEvalStreamingModOCLUnified/Configuration.hpp"
#include "sgpp/datadriven/operation/hash/OperationMultipleEvalStreamingModOCLUnified/OperationMultiEvalStreamingModOCLUnified.hpp"
#include "sgpp/datadriven/operation/hash/OperationMultipleEvalStreamingModOCLUnified/OperatorFactory.hpp"
#include "sgpp/globaldef.hpp"
// #include "autotune/autotune.hpp"
#include "autotune/generalized_kernel.hpp"
#include "autotune/parameter.hpp"
#include "autotune/tuners/bruteforce.hpp"
#include "autotune/tuners/countable_set.hpp"
#include "autotune/tuners/line_search.hpp"
#include "autotune/tuners/neighborhood_search.hpp"

AUTOTUNE_GENERALIZED_KERNEL(void(sgpp::base::DataVector &,
                                 sgpp::base::DataVector &),
                            mult_unified_with_tuning)

AUTOTUNE_GENERALIZED_KERNEL(void(sgpp::base::DataVector &,
                                 sgpp::base::DataVector &),
                            mult_transpose_unified_with_tuning)

namespace sgpp {
namespace datadriven {
namespace StreamingModOCLUnifiedAutoTuneTMP {

/**
 * This class provides an operation for evaluating multiple grid points in the
 * domain and doing least squares data mining. This algorithmic variant uses the
 * streaming algorithm for evaluation. It uses high performance OpenCL kernels
 * and is well-suited for large irregular datasets and grids. This class manages
 * one OpenCL kernel for each devices configured using the
 * OCLOperationConfiguration.
 * When a operation is called it triggers the device work by using OpenMP and
 * delegating the work to instances of the kernels. Furthermore, this class
 * converts the received grid and dataset into a representation that is suited
 * for the streaming algorithm.
 *
 * @see base::OperationMultipleEval
 * @see StreamingModOCLUnified::KernelMult
 * @see StreamingModOCLUnified::KernelMultTranspose
 */
template <typename T>
class OperationMultiEvalStreamingModOCLUnifiedAutoTuneTMP
    : public base::OperationMultipleEval {
protected:
  size_t dims;
  double duration;
  bool verbose;

  std::shared_ptr<base::OCLOperationConfiguration> ocl_parameters_mult;
  std::shared_ptr<base::OCLOperationConfiguration> ocl_parameters_multTranspose;

  std::shared_ptr<sgpp::base::OperationMultipleEval> eval_mult;
  std::shared_ptr<sgpp::base::OperationMultipleEval> eval_multTranspose;

  autotune::countable_set autotune_parameters_mult;
  autotune::countable_set autotune_parameters_multTranspose;

  // has to be enabled during tuning (recompile needed to propagate changed
  // parameters)
  bool always_recompile;

  bool isModLinear;

  int64_t tune_repetitions;

  double duration_mult_acc;
  double duration_multTranspose_acc;

public:
  /**
   * Creates a new instance of the OperationMultiEvalStreamingOCLMultiPlatform
   * class. This class should not be created directly, instead the datadriven
   * operator factory should be used or at least the factory method.
   *
   * @see createStreamingModOCLUnifiedConfigured
   *
   * @param grid The grid to evaluate
   * @param dataset The datapoints to evaluate
   * @param manager The OpenCL manager that manages OpenCL internels for this
   * kernel
   * @param parameters The configuration of the kernel leading to different
   * compute kernels
   */
  OperationMultiEvalStreamingModOCLUnifiedAutoTuneTMP(
      base::Grid &grid, base::DataMatrix &dataset,
      std::shared_ptr<base::OCLOperationConfiguration> parameters,
      bool isModLinear, int64_t tune_repetitions = 1)
      : OperationMultipleEval(grid, dataset), dims(dataset.getNcols()),
        duration(-1.0), always_recompile(true), isModLinear(isModLinear),
        tune_repetitions(tune_repetitions), duration_mult_acc(0.0),
        duration_multTranspose_acc(0.0) {
    this->verbose = (*parameters)["VERBOSE"].getBool();
    this->ocl_parameters_mult =
        std::dynamic_pointer_cast<base::OCLOperationConfiguration>(
            std::shared_ptr<base::OperationConfiguration>(parameters->clone()));
    this->ocl_parameters_multTranspose =
        std::dynamic_pointer_cast<base::OCLOperationConfiguration>(
            std::shared_ptr<base::OperationConfiguration>(parameters->clone()));

    // shared parameters
    autotune::fixed_set_parameter<std::string> p0(
        "OPTIMIZATION_FLAGS",
        {"-cl-unsafe-math-optimizations -cl-denorms-are-zero"}, false);

    // mult/multi-eval parameters
    autotune::fixed_set_parameter<bool> p6("KERNEL_USE_LOCAL_MEMORY",
                                           {true, false});
    autotune::fixed_set_parameter<uint64_t> p7("LOCAL_SIZE", {64, 128, 256});
    autotune::fixed_set_parameter<uint64_t> p1("KERNEL_DATA_BLOCK_SIZE",
                                               {1, 2, 4, 8});
    autotune::fixed_set_parameter<uint64_t> p2("KERNEL_GRID_SPLIT",
                                               {1, 2, 4, 8});
    autotune::fixed_set_parameter<uint64_t> p3("KERNEL_MAX_DIM_UNROLL",
                                               {1, 2, 4, 10});
    autotune::fixed_set_parameter<uint64_t> p4("KERNEL_PREFETCH_SIZE",
                                               {16, 32, 64, 128});
    autotune::fixed_set_parameter<bool> p5("KERNEL_TRANSFER_WHOLE_DATASET",
                                           {true, false});
    autotune::fixed_set_parameter<std::string> p8("KERNEL_STORE_DATA",
                                                  {"array"}, false);

    autotune_parameters_mult.add_parameter(p0);
    autotune_parameters_mult.add_parameter(p1);
    autotune_parameters_mult.add_parameter(p2);
    autotune_parameters_mult.add_parameter(p3);
    autotune_parameters_mult.add_parameter(p4);
    autotune_parameters_mult.add_parameter(p5);
    autotune_parameters_mult.add_parameter(p6);
    autotune_parameters_mult.add_parameter(p7);
    autotune_parameters_mult.add_parameter(p8);

    // trans parameters
    autotune::fixed_set_parameter<bool> p9("KERNEL_TRANS_USE_LOCAL_MEMORY",
                                           {true, false});
    autotune::fixed_set_parameter<uint64_t> p10("TRANS_LOCAL_SIZE",
                                                {64, 128, 256});
    autotune::fixed_set_parameter<uint64_t> p11("KERNEL_TRANS_GRID_BLOCK_SIZE",
                                                {1, 2, 4, 8});
    autotune::fixed_set_parameter<uint64_t> p12("KERNEL_TRANS_DATA_SPLIT",
                                                {1, 2, 4, 8});
    autotune::fixed_set_parameter<uint64_t> p13("KERNEL_TRANS_MAX_DIM_UNROLL",
                                                {1, 2, 4, 10});
    autotune::fixed_set_parameter<uint64_t> p14("KERNEL_TRANS_PREFETCH_SIZE",
                                                {16, 32, 64, 128});
    autotune::fixed_set_parameter<bool> p15("KERNEL_TRANS_TRANSFER_WHOLE_GRID",
                                            {true, false});
    autotune::fixed_set_parameter<std::string> p16("KERNEL_TRANS_STORE_DATA",
                                                   {"array"}, false);

    autotune_parameters_multTranspose.add_parameter(p0);
    autotune_parameters_multTranspose.add_parameter(p9);
    autotune_parameters_multTranspose.add_parameter(p10);
    autotune_parameters_multTranspose.add_parameter(p11);
    autotune_parameters_multTranspose.add_parameter(p12);
    autotune_parameters_multTranspose.add_parameter(p13);
    autotune_parameters_multTranspose.add_parameter(p14);
    autotune_parameters_multTranspose.add_parameter(p15);
    autotune_parameters_multTranspose.add_parameter(p16);

    autotune::mult_unified_with_tuning.set_kernel_functor(
        [this](base::DataVector &alpha, base::DataVector &result) {
          // apply parameters to kernel by re-instantiating
          // notice: of course, this is quite expensive
          // TODO: improve this for more precise tuner runs

          sgpp::datadriven::OperationMultipleEvalConfiguration configuration(
              sgpp::datadriven::OperationMultipleEvalType::STREAMING,
              sgpp::datadriven::OperationMultipleEvalSubType::OCLUNIFIED,
              *(this->ocl_parameters_mult));

          eval_mult = std::shared_ptr<sgpp::base::OperationMultipleEval>(
              datadriven::createStreamingModOCLUnifiedConfigured(
                  this->grid, this->dataset, configuration, this->isModLinear));
          duration_mult_acc = 0.0;
          for (int i = 0; i < this->tune_repetitions + 1; i += 1) {
            eval_mult->mult(alpha, result);
            // always ignore first results
            if (i > 0) {
              duration_mult_acc += this->eval_mult->getDuration();
            }
          }
          duration_mult_acc /= static_cast<double>(this->tune_repetitions);
        });

    autotune::mult_unified_with_tuning.set_kernel_duration_functor(
        [this]() { return this->duration_mult_acc; });

    autotune::mult_unified_with_tuning.set_create_parameter_file_functor(
        [this](autotune::parameter_value_set &parameter_values) {
          this->apply_parameter_values(*this->ocl_parameters_mult,
                                       parameter_values);
        });
    autotune::mult_unified_with_tuning.set_verbose(true);

    autotune::mult_unified_with_tuning.set_valid_parameter_combination_functor(
        [](autotune::parameter_value_set &pv) -> bool {
          if (std::stoull(pv["LOCAL_SIZE"]) <
              std::stoull(pv["KERNEL_PREFETCH_SIZE"])) {
            return false;
          }
          return true;
        });

    autotune::mult_transpose_unified_with_tuning.set_kernel_functor(
        [this](base::DataVector &source, base::DataVector &result) {
          // apply parameters to kernel by re-instantiating
          // notice: of course, this is quite expensive
          // TODO: improve this for more precise tuner runs
          sgpp::datadriven::OperationMultipleEvalConfiguration configuration(
              sgpp::datadriven::OperationMultipleEvalType::STREAMING,
              sgpp::datadriven::OperationMultipleEvalSubType::OCLUNIFIED,
              *(this->ocl_parameters_multTranspose));

          eval_multTranspose =
              std::shared_ptr<sgpp::base::OperationMultipleEval>(
                  datadriven::createStreamingModOCLUnifiedConfigured(
                      this->grid, this->dataset, configuration,
                      this->isModLinear));
          duration_multTranspose_acc = 0.0;
          for (int i = 0; i < this->tune_repetitions + 1; i += 1) {
            eval_multTranspose->multTranspose(source, result);
            // always ignore first results
            if (i > 0) {
              duration_multTranspose_acc +=
                  this->eval_multTranspose->getDuration();
            }
          }
          duration_multTranspose_acc /=
              static_cast<double>(this->tune_repetitions);
        });

    autotune::mult_transpose_unified_with_tuning.set_kernel_duration_functor(
        [this]() { return this->duration_multTranspose_acc; });

    autotune::mult_transpose_unified_with_tuning
        .set_create_parameter_file_functor(
            [this](autotune::parameter_value_set &parameter_values) {
              this->apply_parameter_values(*this->ocl_parameters_multTranspose,
                                           parameter_values);
            });
    autotune::mult_transpose_unified_with_tuning.set_verbose(true);

    autotune::mult_transpose_unified_with_tuning
        .set_valid_parameter_combination_functor(
            [](autotune::parameter_value_set &pv) {
              if (std::stoull(pv["TRANS_LOCAL_SIZE"]) <
                  std::stoull(pv["KERNEL_TRANS_PREFETCH_SIZE"])) {
                return false;
              }
              return true;
            });

    this->prepare();
  }

  /**
   * Destructor
   */
  ~OperationMultiEvalStreamingModOCLUnifiedAutoTuneTMP() {}

  /**
   * Performs the MultiEval operation \f$v:= B^T \alpha\f$.
   *
   * @param alpha The surpluses of the grid
   * @param result A vector that contains the result in the order of the dataset
   */
  void mult(base::DataVector &alpha, base::DataVector &result) override {
    auto start = std::chrono::high_resolution_clock::now();
    autotune::mult_unified_with_tuning(alpha, result);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    duration = diff.count();
  }

  /**
   * Performs the transposed MultiEval operation  \f$v':= B v\f$.
   *
   * @param source The vector \f$v\f$
   * @param result The result of the matrix vector multiplication in the order
   * of grid (of the alpha vector)
   */
  void multTranspose(base::DataVector &source,
                     base::DataVector &result) override {
    auto start = std::chrono::high_resolution_clock::now();
    autotune::mult_transpose_unified_with_tuning(source, result);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    duration = diff.count();
  }

  /**
   * @return The duration of the last call to mult or multTranspose
   */
  double getDuration() override { return this->duration; }

  /**
   * Creates the internal data structures used by the algorithm. Needs to be
   * called every time the grid changes e.g., due to refinement.
   */
  void prepare() override {
    // eval->prepare();
  }

  void tune_mult(base::DataVector &alpha, base::DataVector &result,
                 const std::string &scenario_name,
                 const std::string &tuner_name) {
    // autotune::tuners::bruteforce tuner(autotune::mult_unified_with_tuning,
    // autotune_parameters_mult); tuner.set_write_measurement(scenario_name);
    // tuner.set_verbose(true);

    for (size_t i = 0; i < autotune_parameters_mult.size(); i += 1) {
      autotune_parameters_mult[i]->set_random_value();
    }

    autotune::countable_set optimal_parameters;
    if (tuner_name.compare("bruteforce") == 0) {
      autotune::tuners::bruteforce tuner(autotune::mult_unified_with_tuning,
                                         autotune_parameters_mult);
      tuner.set_verbose(true);
      tuner.set_write_measurement(scenario_name);
      // tuner.setup_test(test_result);
      optimal_parameters = tuner.tune(alpha, result);
    } else if (tuner_name.compare("line_search") == 0) {
      autotune::tuners::line_search tuner(autotune::mult_unified_with_tuning,
                                          autotune_parameters_mult,
                                          10 * autotune_parameters_mult.size());
      tuner.set_verbose(true);
      tuner.set_write_measurement(scenario_name);
      // tuner.setup_test(test_result);
      optimal_parameters = tuner.tune(alpha, result);
    } else if (tuner_name.compare("neighborhood_search") == 0) {
      autotune::tuners::neighborhood_search tuner(
          autotune::mult_unified_with_tuning, autotune_parameters_mult, 100);
      tuner.set_verbose(true);
      tuner.set_write_measurement(scenario_name);
      // tuner.setup_test(test_result);
      optimal_parameters = tuner.tune(alpha, result);
    } else {
      throw "error: tuner not implemented!";
    }
    autotune::mult_unified_with_tuning.set_parameter_values(optimal_parameters);
    autotune::parameter_value_set pv = to_parameter_values(optimal_parameters);
    apply_parameter_values(*(this->ocl_parameters_mult), pv);
    std::stringstream ss;
    this->ocl_parameters_mult->serialize(ss, 0);
    // std::cout << ss.str() << std::endl;
    std::ofstream opt_config(scenario_name + "_optimal.cfg");
    opt_config << ss.str();
    opt_config.close();
  }

  void tune_multTranspose(base::DataVector &source, base::DataVector &result,
                          const std::string &scenario_name,
                          const std::string &tuner_name) {

    for (size_t i = 0; i < autotune_parameters_multTranspose.size(); i += 1) {
      autotune_parameters_multTranspose[i]->set_random_value();
    }

    autotune::countable_set optimal_parameters;
    if (tuner_name.compare("bruteforce") == 0) {
      autotune::tuners::bruteforce tuner(
          autotune::mult_transpose_unified_with_tuning,
          autotune_parameters_multTranspose);
      tuner.set_verbose(true);
      tuner.set_write_measurement(scenario_name);
      // tuner.setup_test(test_result);
      optimal_parameters = tuner.tune(source, result);
    } else if (tuner_name.compare("line_search") == 0) {
      autotune::tuners::line_search tuner(
          autotune::mult_transpose_unified_with_tuning,
          autotune_parameters_multTranspose,
          10 * autotune_parameters_multTranspose.size());
      tuner.set_verbose(true);
      tuner.set_write_measurement(scenario_name);
      // tuner.setup_test(test_result);
      optimal_parameters = tuner.tune(source, result);
    } else if (tuner_name.compare("neighborhood_search") == 0) {
      autotune::tuners::neighborhood_search tuner(
          autotune::mult_transpose_unified_with_tuning,
          autotune_parameters_multTranspose, 100);
      tuner.set_verbose(true);
      tuner.set_write_measurement(scenario_name);
      // tuner.setup_test(test_result);
      optimal_parameters = tuner.tune(source, result);
    } else {
      throw "error: tuner not implemented!";
    }
    autotune::mult_transpose_unified_with_tuning.set_parameter_values(
        optimal_parameters);
    autotune::parameter_value_set pv = to_parameter_values(optimal_parameters);
    apply_parameter_values(*(this->ocl_parameters_multTranspose), pv);
    std::stringstream ss;
    this->ocl_parameters_multTranspose->serialize(ss, 0);
    // std::cout << ss.str() << std::endl;
    std::ofstream opt_config(scenario_name + "_optimal.cfg");
    opt_config << ss.str();
    opt_config.close();
  }

  void apply_parameter_values(base::OCLOperationConfiguration &config,
                              autotune::parameter_value_set &parameter_values) {
    for (std::string &platformName : config["PLATFORMS"].keys()) {
      json::node &platformNode = config["PLATFORMS"][platformName];
      for (std::string &deviceName : platformNode["DEVICES"].keys()) {
        json::node &deviceNode = platformNode["DEVICES"][deviceName];

        const std::string &kernelName = sgpp::datadriven::
            StreamingModOCLUnified::Configuration::getKernelName();
        json::node &kernelNode =
            deviceNode["KERNELS"].contains(kernelName)
                ? deviceNode["KERNELS"][kernelName]
                : deviceNode["KERNELS"].addDictAttr(kernelName);
        for (auto &p : parameter_values) {
          std::cout << "parameter name: " << p.first << " value: " << p.second
                    << std::endl;
          kernelNode.replaceTextAttr(p.first, p.second);
          kernelNode.replaceTextAttr("VERBOSE", "true");
        }
      }
    }

    // std::stringstream ss;
    // config.serialize(ss, 0);
    // std::cout << ss.str() << std::endl;
  }
};

} // namespace StreamingModOCLUnifiedAutoTuneTMP
} // namespace datadriven
} // namespace sgpp
