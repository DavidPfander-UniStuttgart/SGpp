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
#include "KernelCreateGraph.hpp"
namespace sgpp {
namespace datadriven {

std::unique_ptr<DensityOCLMultiPlatform::OperationCreateGraphOCL> createNearestNeighborGraphConfigured(
    base::DataMatrix &dataset, size_t k, size_t dimensions, std::string opencl_conf) {
  std::cout << "Using configuration file " << opencl_conf << std::endl;
  auto parameters = std::make_shared<sgpp::base::OCLOperationConfiguration>(opencl_conf);
  auto manager = std::make_shared<base::OCLManagerMultiPlatform>(parameters);
  DensityOCLMultiPlatform::OperationCreateGraphOCL::load_default_parameters(parameters);

  if ((*parameters)["INTERNAL_PRECISION"].get().compare("float") == 0) {
    return std::make_unique<DensityOCLMultiPlatform::OperationCreateGraphOCLSingleDevice<float>>(
        dataset, dimensions, manager, parameters, k);
  } else if ((*parameters)["INTERNAL_PRECISION"].get().compare("double") == 0) {
    return std::make_unique<DensityOCLMultiPlatform::OperationCreateGraphOCLSingleDevice<double>>(
        dataset, dimensions, manager, parameters, k);
  } else {
    std::stringstream errorString;
    errorString << "Error creating operation\"CreateGraphOCL\": "
                << " invalid value for parameter \"INTERNAL_PRECISION\"";
    throw base::factory_exception(errorString.str().c_str());
  }
  return nullptr;
}
std::unique_ptr<DensityOCLMultiPlatform::OperationCreateGraphOCL> createNearestNeighborGraphConfigured(
    double *dataset, size_t dataset_size, size_t k, size_t dimensions,
    std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters) {
  DensityOCLMultiPlatform::OperationCreateGraphOCL::load_default_parameters(parameters);
  auto manager = std::make_shared<base::OCLManagerMultiPlatform>(parameters);

  if ((*parameters)["INTERNAL_PRECISION"].get().compare("float") == 0) {
    return std::make_unique<DensityOCLMultiPlatform::OperationCreateGraphOCLSingleDevice<float>>(
        dataset, dataset_size, dimensions, manager, parameters, k);
  } else if ((*parameters)["INTERNAL_PRECISION"].get().compare("double") == 0) {
    return std::make_unique<DensityOCLMultiPlatform::OperationCreateGraphOCLSingleDevice<double>>(
        dataset, dataset_size, dimensions, manager, parameters, k);
  } else {
    std::stringstream errorString;
    errorString << "Error creating operation\"CreateGraphOCL\": "
                << " invalid value for parameter \"INTERNAL_PRECISION\"";
    throw base::factory_exception(errorString.str().c_str());
  }
  return nullptr;
}

std::unique_ptr<DensityOCLMultiPlatform::OperationCreateGraphOCL> createNearestNeighborGraphConfigured(
    double *dataset, size_t dataset_size, size_t k, size_t dimensions, std::string opencl_conf) {
  std::cout << "Using configuration file " << opencl_conf << std::endl;
  auto parameters = std::make_shared<sgpp::base::OCLOperationConfiguration>(opencl_conf);
  auto manager = std::make_shared<base::OCLManagerMultiPlatform>(parameters);
  DensityOCLMultiPlatform::OperationCreateGraphOCL::load_default_parameters(parameters);

  if ((*parameters)["INTERNAL_PRECISION"].get().compare("float") == 0) {
    return std::make_unique<DensityOCLMultiPlatform::OperationCreateGraphOCLSingleDevice<float>>(
        dataset, dataset_size, dimensions, manager, parameters, k);
  } else if ((*parameters)["INTERNAL_PRECISION"].get().compare("double") == 0) {
    return std::make_unique<DensityOCLMultiPlatform::OperationCreateGraphOCLSingleDevice<double>>(
        dataset, dataset_size, dimensions, manager, parameters, k);
  } else {
    std::stringstream errorString;
    errorString << "Error creating operation\"CreateGraphOCL\": "
                << " invalid value for parameter \"INTERNAL_PRECISION\"";
    throw base::factory_exception(errorString.str().c_str());
  }
  return nullptr;
}
}  // namespace datadriven
}  // namespace sgpp
