// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#pragma once
#include <sgpp/base/exception/operation_exception.hpp>
#include <sgpp/base/grid/GridStorage.hpp>
#include <sgpp/base/opencl/OCLManager.hpp>
#include <sgpp/base/opencl/OCLOperationConfiguration.hpp>
#include <sgpp/base/operation/hash/OperationMatrix.hpp>
#include <sgpp/base/tools/SGppStopwatch.hpp>
#include "KernelDensityB.hpp"
#include "KernelDensityMult.hpp"

namespace sgpp {
namespace datadriven {
namespace DensityOCLMultiPlatform {

/// Base class for density multiplication operation
class OperationDensity : public base::OperationMatrix {
 protected:
  double last_duration_b;
  double last_duration_density;
  double acc_duration_b;
  double acc_duration_density;

 public:
  OperationDensity()
      : last_duration_b(-1.0),
        last_duration_density(-1.0),
        acc_duration_b(0.0),
        acc_duration_density(0.0) {}
  /// Execute one matrix-vector multiplication with the density matrix
  virtual void mult(base::DataVector &alpha, base::DataVector &result) = 0;
  /// Use before calling partial_mult directly
  virtual void initialize_alpha(std::vector<double> &alpha) = 0;
  /// Execute a partial (startindex to startindex+chunksize) multiplication with the density matrix
  virtual void start_partial_mult(int start_id, int chunksize) = 0;
  virtual void finish_partial_mult(double *result, int start_id, int chunksize) = 0;
  /// Generates the right hand side vector for the density equation
  virtual void generateb(base::DataMatrix &dataset, sgpp::base::DataVector &b, size_t start_id = 0,
                         size_t chunksize = 0) = 0;
  virtual void initialize_dataset(base::DataMatrix &dataset) = 0;
  virtual void start_rhs_generation(size_t start_id, size_t chunksize) = 0;
  virtual void finalize_rhs_generation(sgpp::base::DataVector &b, size_t start_id,
                                       size_t chunksize) = 0;

  double getLastDurationDensityMult() { return last_duration_density; }

  double getLastDurationB() { return last_duration_b; }

  void resetAccDurationDensityMult() { acc_duration_density = 0.0; }

  void resetAccDurationB() { acc_duration_b = 0.0; }

  double getAccDurationDensityMult() { return acc_duration_density; }

  double getAccDurationB() { return acc_duration_b; }

  /// Generate the default parameters in die json configuration
  static void load_default_parameters(std::shared_ptr<base::OCLOperationConfiguration> parameters) {
    if (parameters->contains("INTERNAL_PRECISION") == false) {
      std::cout << "Warning! No internal precision setting detected."
                << " Using double precision from now on!" << std::endl;
      parameters->addIDAttr("INTERNAL_PRECISION", "double");
    }

    if ((*parameters)["INTERNAL_PRECISION"].get().compare("float") == 0) {
      DensityOCLMultiPlatform::KernelDensityMultInterface<float>::augmentDefaultParameters(*parameters);
      DensityOCLMultiPlatform::KernelDensityBInterface<float>::augmentDefaultParameters(*parameters);
    } else if ((*parameters)["INTERNAL_PRECISION"].get().compare("double") == 0) {
      DensityOCLMultiPlatform::KernelDensityMultInterface<double>::augmentDefaultParameters(*parameters);
      DensityOCLMultiPlatform::KernelDensityBInterface<double>::augmentDefaultParameters(*parameters);
    } else {
      std::stringstream errorString;
      errorString << "Error creating operation\"OperationDensityOCLMultiPlatform\": "
                  << " invalid value for parameter \"INTERNAL_PRECISION\"";
      throw base::operation_exception(errorString.str().c_str());
    }
  }
};

}  // namespace DensityOCLMultiPlatform
}  // namespace datadriven
}  // namespace sgpp
