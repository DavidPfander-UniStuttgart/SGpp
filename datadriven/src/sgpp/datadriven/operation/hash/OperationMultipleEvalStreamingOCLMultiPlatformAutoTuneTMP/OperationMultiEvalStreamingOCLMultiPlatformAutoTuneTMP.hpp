// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#pragma once

#include <algorithm>
#include <chrono>
#include <mutex>  // NOLINT(build/c++11)
#include <sstream>
#include <vector>

#include "sgpp/base/exception/operation_exception.hpp"
#include "sgpp/base/opencl/OCLManagerMultiPlatform.hpp"
#include "sgpp/base/opencl/OCLOperationConfiguration.hpp"
#include "sgpp/base/operation/hash/OperationMultipleEval.hpp"
#include "sgpp/datadriven/DatadrivenOpFactory.hpp"
#include "sgpp/datadriven/operation/hash/DatadrivenOperationCommon.hpp"
#include "sgpp/datadriven/operation/hash/OperationMultipleEvalStreamingOCLMultiPlatform/Configuration.hpp"
#include "sgpp/datadriven/operation/hash/OperationMultipleEvalStreamingOCLMultiPlatform/OperationMultiEvalStreamingOCLMultiPlatform.hpp"
#include "sgpp/datadriven/operation/hash/OperationMultipleEvalStreamingOCLMultiPlatform/OperatorFactory.hpp"
#include "sgpp/globaldef.hpp"
// #include "autotune/autotune.hpp"
#include "autotune/generalized_kernel.hpp"
#include "autotune/parameter.hpp"
#include "autotune/tuners/bruteforce.hpp"
#include "autotune/tuners/countable_set.hpp"

AUTOTUNE_DECLARE_DEFINE_GENERALIZED_KERNEL(void(sgpp::base::DataVector &, sgpp::base::DataVector &),
                                           mult_with_tuning)

AUTOTUNE_DECLARE_DEFINE_GENERALIZED_KERNEL(void(sgpp::base::DataVector &, sgpp::base::DataVector &),
                                           mult_transpose_with_tuning)

namespace sgpp {
namespace datadriven {
namespace StreamingOCLMultiPlatformAutoTuneTMP {

/**
 * This class provides an operation for evaluating multiple grid points in the domain and doing
 * least squares data mining.
 * This algorithmic variant uses the streaming algorithm for evaluation.
 * It uses high performance OpenCL kernels and is well-suited for large irregular datasets and
 * grids.
 * This class manages one OpenCL kernel for each devices configured using the
 * OCLOperationConfiguration.
 * When a operation is called it triggers the device work by using OpenMP and delegating the work to
 * instances of the kernels.
 * Furthermore, this class converts the received grid and dataset into a representation that is
 * suited for the streaming algorithm.
 *
 * @see base::OperationMultipleEval
 * @see StreamingOCLMultiPlatform::KernelMult
 * @see StreamingOCLMultiPlatform::KernelMultTranspose
 */
template <typename T>
class OperationMultiEvalStreamingOCLMultiPlatformAutoTuneTMP : public base::OperationMultipleEval {
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

 public:
  /**
   * Creates a new instance of the OperationMultiEvalStreamingOCLMultiPlatform class.
   * This class should not be created directly, instead the datadriven operator factory should
   * be
   * used or at least the factory method.
   *
   * @see createStreamingOCLMultiPlatformConfigured
   *
   * @param grid The grid to evaluate
   * @param dataset The datapoints to evaluate
   * @param manager The OpenCL manager that manages OpenCL internels for this kernel
   * @param parameters The configuration of the kernel leading to different compute kernels
   */
  OperationMultiEvalStreamingOCLMultiPlatformAutoTuneTMP(
      base::Grid &grid, base::DataMatrix &dataset,
      std::shared_ptr<base::OCLOperationConfiguration> parameters)
      : OperationMultipleEval(grid, dataset), duration(-1.0) {
    this->dims = dataset.getNcols();  // be aware of transpose!
    this->verbose = (*parameters)["VERBOSE"].getBool();
    this->ocl_parameters_mult = std::dynamic_pointer_cast<base::OCLOperationConfiguration>(
        std::shared_ptr<base::OperationConfiguration>(parameters->clone()));
    this->ocl_parameters_multTranspose = std::dynamic_pointer_cast<base::OCLOperationConfiguration>(
        std::shared_ptr<base::OperationConfiguration>(parameters->clone()));

    autotune::fixed_set_parameter<int64_t> p1("LOCAL_SIZE", {64, 128, 256});
    autotune::fixed_set_parameter<bool> p2("KERNEL_USE_LOCAL_MEMORY", {true, false});
    autotune::fixed_set_parameter<std::string> p3("KERNEL_STORE_DATA", {"array"}, false);
    autotune::fixed_set_parameter<int64_t> p4("KERNEL_MAX_DIM_UNROLL", {1, 10});
    autotune::fixed_set_parameter<int64_t> p5("KERNEL_DATA_BLOCK_SIZE", {1, 2, 4});
    // autotune::fixed_set_parameter p6("KERNEL_TRANS_GRID_BLOCK_SIZE", {1, 2, 4});
    autotune::fixed_set_parameter<int64_t> p7("KERNEL_SCHEDULE_SIZE", {102400});
    autotune::fixed_set_parameter<int64_t> p8("KERNEL_PREFETCH_SIZE", {32, 64, 128});
    // autotune::fixed_set_parameter p9("KERNEL_TRANS_PREFETCH_SIZE", {32, 64, 128});

    autotune_parameters_mult.add_parameter(p1);
    autotune_parameters_mult.add_parameter(p2);
    autotune_parameters_mult.add_parameter(p3);
    autotune_parameters_mult.add_parameter(p4);
    autotune_parameters_mult.add_parameter(p5);
    // autotune_parameters_mult.add_parameter(p6);
    autotune_parameters_mult.add_parameter(p7);
    autotune_parameters_mult.add_parameter(p8);
    // autotune_parameters_mult.add_parameter(p9);

    autotune_parameters_multTranspose.add_parameter(p1);
    autotune_parameters_multTranspose.add_parameter(p2);
    autotune_parameters_multTranspose.add_parameter(p3);
    autotune_parameters_multTranspose.add_parameter(p4);
    autotune_parameters_multTranspose.add_parameter(p5);
    // autotune_parameters_multTranspose.add_parameter(p6);
    autotune_parameters_multTranspose.add_parameter(p7);
    autotune_parameters_multTranspose.add_parameter(p8);
    // autotune_parameters_multTranspose.add_parameter(p9);

    autotune::mult_with_tuning.set_kernel_functor(
        [this](base::DataVector &alpha, base::DataVector &result) {
          // apply parameters to kernel by re-instantiating
          // notice: of course, this is quite expensive
          // TODO: improve this for more precise tuner runs
          sgpp::datadriven::OperationMultipleEvalConfiguration configuration(
              sgpp::datadriven::OperationMultipleEvalType::STREAMING,
              sgpp::datadriven::OperationMultipleEvalSubType::OCLMP, *(this->ocl_parameters_mult));

          eval_mult = std::shared_ptr<sgpp::base::OperationMultipleEval>(
              datadriven::createStreamingOCLMultiPlatformConfigured(this->grid, this->dataset,
                                                                    configuration));
          eval_mult->mult(alpha, result);
        });

    autotune::mult_with_tuning.set_kernel_duration_functor(
        [this]() { return this->eval_mult->getDuration(); });

    autotune::mult_with_tuning.set_create_parameter_file_functor(
        [this](autotune::parameter_value_set &parameter_values) {
          for (auto &p : parameter_values) {
            for (std::string &platformName : (*this->ocl_parameters_mult)["PLATFORMS"].keys()) {
              json::Node &platformNode = (*this->ocl_parameters_mult)["PLATFORMS"][platformName];
              for (std::string &deviceName : platformNode["DEVICES"].keys()) {
                json::Node &deviceNode = platformNode["DEVICES"][deviceName];

                const std::string &kernelName =
                    sgpp::datadriven::StreamingOCLMultiPlatform::Configuration::getKernelName();
                json::Node &kernelNode = deviceNode["KERNELS"].contains(kernelName)
                                             ? deviceNode["KERNELS"][kernelName]
                                             : deviceNode["KERNELS"].addDictAttr(kernelName);
                // json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
                std::cout << "parameter name: " << p.first << " value: " << p.second << std::endl;
                kernelNode.replaceTextAttr(p.first, p.second);
                kernelNode.replaceTextAttr("VERBOSE", "true");
              }
            }
          }
          std::stringstream ss;
          this->ocl_parameters_mult->serialize(ss, 0);
          std::cout << ss.str() << std::endl;
        });
    autotune::mult_with_tuning.set_verbose(true);

    autotune::mult_transpose_with_tuning.set_kernel_functor(
        [this](base::DataVector &source, base::DataVector &result) {
          // apply parameters to kernel by re-instantiating
          // notice: of course, this is quite expensive
          // TODO: improve this for more precise tuner runs
          sgpp::datadriven::OperationMultipleEvalConfiguration configuration(
              sgpp::datadriven::OperationMultipleEvalType::STREAMING,
              sgpp::datadriven::OperationMultipleEvalSubType::OCLMP,
              *(this->ocl_parameters_multTranspose));

          eval_multTranspose = std::shared_ptr<sgpp::base::OperationMultipleEval>(
              datadriven::createStreamingOCLMultiPlatformConfigured(this->grid, this->dataset,
                                                                    configuration));
          eval_multTranspose->multTranspose(source, result);
        });

    autotune::mult_transpose_with_tuning.set_kernel_duration_functor(
        [this]() { return this->eval_mult->getDuration(); });

    autotune::mult_transpose_with_tuning.set_create_parameter_file_functor(
        [this](autotune::parameter_value_set &parameter_values) {
          for (auto &p : parameter_values) {
            for (std::string &platformName :
                 (*this->ocl_parameters_multTranspose)["PLATFORMS"].keys()) {
              json::Node &platformNode =
                  (*this->ocl_parameters_multTranspose)["PLATFORMS"][platformName];
              for (std::string &deviceName : platformNode["DEVICES"].keys()) {
                json::Node &deviceNode = platformNode["DEVICES"][deviceName];

                const std::string &kernelName =
                    sgpp::datadriven::StreamingOCLMultiPlatform::Configuration::getKernelName();
                json::Node &kernelNode = deviceNode["KERNELS"].contains(kernelName)
                                             ? deviceNode["KERNELS"][kernelName]
                                             : deviceNode["KERNELS"].addDictAttr(kernelName);
                // json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
                std::cout << "parameter name: " << p.first << " value: " << p.second << std::endl;
                kernelNode.replaceTextAttr(p.first, p.second);
                kernelNode.replaceTextAttr("VERBOSE", "true");
              }
            }
          }
          std::stringstream ss;
          this->ocl_parameters_multTranspose->serialize(ss, 0);
          std::cout << ss.str() << std::endl;
        });
    autotune::mult_transpose_with_tuning.set_verbose(true);

    this->prepare();
  }

  /**
   * Destructor
   */
  ~OperationMultiEvalStreamingOCLMultiPlatformAutoTuneTMP() {}

  /**
   * Performs the MultiEval operation \f$v:= B^T \alpha\f$.
   *
   * @param alpha The surpluses of the grid
   * @param result A vector that contains the result in the order of the dataset
   */
  void mult(base::DataVector &alpha, base::DataVector &result) override {
    auto start = std::chrono::high_resolution_clock::now();
    autotune::mult_with_tuning(alpha, result);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    duration = diff.count();
  }

  /**
   * Performs the transposed MultiEval operation  \f$v':= B v\f$.
   *
   * @param source The vector \f$v\f$
   * @param result The result of the matrix vector multiplication in the order of grid (of the
   * alpha
   * vector)
   */
  void multTranspose(base::DataVector &source, base::DataVector &result) override {
    auto start = std::chrono::high_resolution_clock::now();
    autotune::mult_transpose_with_tuning(source, result);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    duration = diff.count();
  }

  /**
   * @return The duration of the last call to mult or multTranspose
   */
  double getDuration() override { return this->duration; }

  /**
   * Creates the internal data structures used by the algorithm. Needs to be called every time
   * the
   * grid changes e.g., due to refinement.
   */
  void prepare() override {
    // eval->prepare();
  }

  void tune_mult(base::DataVector &alpha, base::DataVector &result) {
    autotune::mult_with_tuning.set_write_measurement("mult_bruteforce");
    autotune::tuners::bruteforce tuner(autotune::mult_with_tuning, autotune_parameters_mult);
    tuner.set_verbose(true);

    autotune::countable_set optimal_parameters = tuner.tune(alpha, result);
    autotune::mult_with_tuning.set_parameter_values(optimal_parameters);
  }

  void tune_multTranspose(base::DataVector &alpha, base::DataVector &result) {
    autotune::mult_transpose_with_tuning.set_write_measurement("multTranspose_bruteforce");
    autotune::tuners::bruteforce tuner(autotune::mult_with_tuning,
                                       autotune_parameters_multTranspose);
    tuner.set_verbose(true);

    autotune::countable_set optimal_parameters = tuner.tune(alpha, result);
    autotune::mult_transpose_with_tuning.set_parameter_values(optimal_parameters);
  }
};

}  // namespace StreamingOCLMultiPlatformAutoTuneTMP
}  // namespace datadriven
}  // namespace sgpp
