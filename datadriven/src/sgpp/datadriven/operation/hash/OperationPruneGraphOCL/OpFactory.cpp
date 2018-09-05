// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#include "OpFactory.hpp"
#include <sgpp/base/exception/factory_exception.hpp>
#include <sgpp/base/opencl/OCLManager.hpp>
#include <sgpp/base/opencl/OCLOperationConfiguration.hpp>
#include <sgpp/globaldef.hpp>
#include <string>
#include "OperationPruneGraphOCL.hpp"
namespace sgpp {
namespace datadriven {

DensityOCLMultiPlatform::OperationPruneGraphOCL *pruneNearestNeighborGraphConfigured(
    base::Grid &grid, size_t dimensions, base::DataVector &alpha, base::DataMatrix &data,
    double threshold, size_t k, std::string opencl_conf) {
  std::cout << "Using configuration file " << opencl_conf << std::endl;
  auto parameters = std::make_shared<sgpp::base::OCLOperationConfiguration>(opencl_conf);
  DensityOCLMultiPlatform::OperationPruneGraphOCL::load_default_parameters(parameters);
  auto manager = std::make_shared<base::OCLManagerMultiPlatform>(parameters);

  if ((*parameters)["INTERNAL_PRECISION"].get().compare("float") == 0) {
    return new DensityOCLMultiPlatform::OperationPruneGraphOCLMultiPlatform<float>(
        grid, alpha, data, dimensions, manager, parameters, static_cast<float>(threshold), k);
  } else if ((*parameters)["INTERNAL_PRECISION"].get().compare("double") == 0) {
    return new DensityOCLMultiPlatform::OperationPruneGraphOCLMultiPlatform<double>(
        grid, alpha, data, dimensions, manager, parameters, threshold, k);
  } else {
    std::stringstream errorString;
    errorString << "Error creating operation\"OperationPruneGraphOCL\": "
                << " invalid value for parameter \"INTERNAL_PRECISION\"";
    throw base::factory_exception(errorString.str().c_str());
  }
  return nullptr;
}
DensityOCLMultiPlatform::OperationPruneGraphOCL *pruneNearestNeighborGraphConfigured(
    const std::vector<int> &gridpoints, size_t gridsize, size_t dimensions,
    std::vector<double> &alpha, base::DataMatrix &data,
    double threshold, size_t k, std::string opencl_conf) {
  std::cout << "Using configuration file " << opencl_conf << std::endl;
  auto parameters = std::make_shared<sgpp::base::OCLOperationConfiguration>(opencl_conf);
  DensityOCLMultiPlatform::OperationPruneGraphOCL::load_default_parameters(parameters);
  auto manager = std::make_shared<base::OCLManagerMultiPlatform>(parameters);

  if ((*parameters)["INTERNAL_PRECISION"].get().compare("float") == 0) {
    return new DensityOCLMultiPlatform::OperationPruneGraphOCLMultiPlatform<float>(
        gridpoints, gridsize, dimensions, alpha, data, manager, parameters,
        static_cast<float>(threshold), k);
  } else if ((*parameters)["INTERNAL_PRECISION"].get().compare("double") == 0) {
    return new DensityOCLMultiPlatform::OperationPruneGraphOCLMultiPlatform<double>(
        gridpoints, gridsize, dimensions, alpha, data, manager, parameters, threshold, k);
  } else {
    std::stringstream errorString;
    errorString << "Error creating operation\"OperationPruneGraphOCL\": "
                << " invalid value for parameter \"INTERNAL_PRECISION\"";
    throw base::factory_exception(errorString.str().c_str());
  }
  return nullptr;
}

DensityOCLMultiPlatform::OperationPruneGraphOCL *pruneNearestNeighborGraphConfigured(
    const std::vector<int> &gridpoints, size_t gridsize, size_t dimensions,
    std::vector<double> &alpha, base::DataMatrix &data,
    double threshold, size_t k, std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters) {
  auto manager = std::make_shared<base::OCLManagerMultiPlatform>(parameters);
  DensityOCLMultiPlatform::OperationPruneGraphOCL::load_default_parameters(parameters);

  if ((*parameters)["INTERNAL_PRECISION"].get().compare("float") == 0) {
    return new DensityOCLMultiPlatform::OperationPruneGraphOCLMultiPlatform<float>(
        gridpoints, gridsize, dimensions, alpha, data, manager, parameters,
        static_cast<float>(threshold), k);
  } else if ((*parameters)["INTERNAL_PRECISION"].get().compare("double") == 0) {
    return new DensityOCLMultiPlatform::OperationPruneGraphOCLMultiPlatform<double>(
        gridpoints, gridsize, dimensions, alpha, data, manager, parameters, threshold, k);
  } else {
    std::stringstream errorString;
    errorString << "Error creating operation\"OperationPruneGraphOCL\": "
                << " invalid value for parameter \"INTERNAL_PRECISION\"";
    throw base::factory_exception(errorString.str().c_str());
  }
  return nullptr;
}
}  // namespace datadriven
}  // namespace sgpp
