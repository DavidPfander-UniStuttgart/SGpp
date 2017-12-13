// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#pragma once

#include <algorithm>
#include <chrono>
#include <mutex>  // NOLINT(build/c++11)
#include <vector>

#include "sgpp/base/exception/operation_exception.hpp"
#include "sgpp/base/opencl/OCLManagerMultiPlatform.hpp"
#include "sgpp/base/opencl/OCLOperationConfiguration.hpp"
#include "sgpp/base/operation/hash/OperationMultipleEval.hpp"
#include "sgpp/globaldef.hpp"

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
class OperationMultiEvalStreamingOCLMultiPlatform : public base::OperationMultipleEval {
 protected:
  size_t dims;
  double duration;
  bool verbose;

  std::shared_ptr<base::OCLManagerMultiPlatform> manager;
  std::shared_ptr<base::OCLOperationConfiguration> parameters;

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
  OperationMultiEvalStreamingOCLMultiPlatform(
      base::Grid &grid, base::DataMatrix &dataset,
      std::shared_ptr<base::OCLManagerMultiPlatform> manager,
      std::shared_ptr<base::OCLOperationConfiguration> parameters)
      : OperationMultipleEval(grid, dataset),
        duration(-1.0),
        manager(manager),
        parameters(parameters) {
    this->dims = dataset.getNcols();  // be aware of transpose!
    this->verbose = (*parameters)["VERBOSE"].getBool();

    this->prepare();
  }

  /**
   * Destructor
   */
  ~OperationMultiEvalStreamingOCLMultiPlatform() {}

  /**
   * Performs the MultiEval operation \f$v:= B^T \alpha\f$.
   *
   * @param alpha The surpluses of the grid
   * @param result A vector that contains the result in the order of the dataset
   */
  void mult(base::DataVector &alpha, base::DataVector &result) override {}

  /**
   * Performs the transposed MultiEval operation  \f$v':= B v\f$.
   *
   * @param source The vector \f$v\f$
   * @param result The result of the matrix vector multiplication in the order of grid (of the alpha
   * vector)
   */
  void multTranspose(base::DataVector &source, base::DataVector &result) override {}

  /**
   * @return The duration of the last call to mult or multTranspose
   */
  double getDuration() override { return this->duration; }

  /**
   * Creates the internal data structures used by the algorithm. Needs to be called every time the
   * grid changes e.g., due to refinement.
   */
  void prepare() override {}
};

}  // namespace StreamingOCLMultiPlatformAutoTuneTMP
}  // namespace datadriven
}  // namespace sgpp
