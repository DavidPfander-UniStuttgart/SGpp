// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#pragma once
#include <sgpp/base/operation/hash/OperationMatrix.hpp>
#include <sgpp/base/grid/GridStorage.hpp>
#include <sgpp/base/tools/SGppStopwatch.hpp>
#include <sgpp/base/exception/operation_exception.hpp>
#include <sgpp/base/opencl/OCLOperationConfiguration.hpp>
#include <sgpp/base/opencl/OCLManager.hpp>
#include "KernelMult.hpp"
#include "KernelB.hpp"

namespace sgpp {
namespace datadriven {
namespace DensityOCLMultiPlatform {

class OperationDensityOCL: public base::OperationMatrix {
 public:
  OperationDensityOCL()  {
  }
  virtual void mult(base::DataVector& alpha, base::DataVector& result) = 0;
  virtual void partial_mult(double *alpha, double *result, size_t start_id, size_t chunksize) = 0;
  virtual void generateb(base::DataMatrix &dataset, sgpp::base::DataVector &b,
                         size_t start_id = 0,  size_t chunksize = 0) = 0;
  static void load_default_parameters(base::OCLOperationConfiguration *parameters) {
  if (parameters->contains("INTERNAL_PRECISION") == false) {
    std::cout << "Warning! No internal precision setting detected."
              << " Using double precision from now on!" << std::endl;
    parameters->addIDAttr("INTERNAL_PRECISION", "double");
  }

  if ((*parameters)["INTERNAL_PRECISION"].get().compare("float") == 0) {
    DensityOCLMultiPlatform::KernelDensityMult<float>::augmentDefaultParameters(*parameters);
    DensityOCLMultiPlatform::KernelDensityB<float>::augmentDefaultParameters(*parameters);
  } else if ((*parameters)["INTERNAL_PRECISION"].get().compare("double") == 0) {
    DensityOCLMultiPlatform::KernelDensityMult<double>::augmentDefaultParameters(*parameters);
    DensityOCLMultiPlatform::KernelDensityB<double>::augmentDefaultParameters(*parameters);
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