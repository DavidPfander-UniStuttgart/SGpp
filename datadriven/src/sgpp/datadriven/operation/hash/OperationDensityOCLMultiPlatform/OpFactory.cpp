// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org
#include <sgpp/base/exception/factory_exception.hpp>
#include <sgpp/base/grid/Grid.hpp>
#include <sgpp/base/opencl/OCLManager.hpp>
#include <sgpp/base/opencl/OCLOperationConfiguration.hpp>
#include <sgpp/globaldef.hpp>
#include <string>

#include "OperationDensityOCLMultiPlatform.hpp"
namespace sgpp {
namespace datadriven {
std::unique_ptr<DensityOCLMultiPlatform::OperationDensity> createDensityOCLMultiPlatformConfigured(
    base::Grid& grid, size_t dimension, double lambda,
    std::shared_ptr<base::OCLOperationConfiguration> parameters) {
  auto manager = std::make_shared<base::OCLManagerMultiPlatform>(parameters);

  DensityOCLMultiPlatform::OperationDensity::load_default_parameters(parameters);
  if ((*parameters)["INTERNAL_PRECISION"].get().compare("float") == 0) {
    return std::make_unique<DensityOCLMultiPlatform::OperationDensityOCLMultiPlatform<float>>(
        grid, dimension, manager, parameters, static_cast<float>(lambda));
  } else if ((*parameters)["INTERNAL_PRECISION"].get().compare("double") == 0) {
    return std::make_unique<DensityOCLMultiPlatform::OperationDensityOCLMultiPlatform<double>>(
        grid, dimension, manager, parameters, lambda);
  } else {
    std::stringstream errorString;
    errorString << "Error creating operation\"OperationDensityOCLMultiPlatform\": "
                << " invalid value for parameter \"INTERNAL_PRECISION\"";
    throw base::factory_exception(errorString.str().c_str());
  }
  return nullptr;
}
std::unique_ptr<DensityOCLMultiPlatform::OperationDensity> createDensityOCLMultiPlatformConfigured(
    base::Grid& grid, size_t dimension, double lambda, std::string opencl_conf) {
  std::cout << "Using configuration file " << opencl_conf << std::endl;
  auto parameters = std::make_shared<base::OCLOperationConfiguration>(opencl_conf);
  DensityOCLMultiPlatform::OperationDensity::load_default_parameters(parameters);
  auto manager = std::make_shared<base::OCLManagerMultiPlatform>(parameters);

  if ((*parameters)["INTERNAL_PRECISION"].get().compare("float") == 0) {
    return std::make_unique<DensityOCLMultiPlatform::OperationDensityOCLMultiPlatform<float>>(
        grid, dimension, manager, parameters, static_cast<float>(lambda));
  } else if ((*parameters)["INTERNAL_PRECISION"].get().compare("double") == 0) {
    return std::make_unique<DensityOCLMultiPlatform::OperationDensityOCLMultiPlatform<double>>(
        grid, dimension, manager, parameters, lambda);
  } else {
    std::stringstream errorString;
    errorString << "Error creating operation\"OperationDensityOCLMultiPlatform\": "
                << " invalid value for parameter \"INTERNAL_PRECISION\"";
    throw base::factory_exception(errorString.str().c_str());
  }
  return nullptr;
}
std::unique_ptr<DensityOCLMultiPlatform::OperationDensity> createDensityOCLMultiPlatformConfigured(
    const std::vector<int> &gridpoints, size_t gridsize, size_t dimension, double lambda, std::string opencl_conf) {
  std::cout << "Using configuration file " << opencl_conf << std::endl;
  auto parameters = std::make_shared<base::OCLOperationConfiguration>(opencl_conf);
  DensityOCLMultiPlatform::OperationDensity::load_default_parameters(parameters);
  auto manager = std::make_shared<base::OCLManagerMultiPlatform>(parameters);

  if ((*parameters)["INTERNAL_PRECISION"].get().compare("float") == 0) {
    return std::make_unique<DensityOCLMultiPlatform::OperationDensityOCLMultiPlatform<float>>(
        gridpoints, gridsize, dimension, manager, parameters, static_cast<float>(lambda));
  } else if ((*parameters)["INTERNAL_PRECISION"].get().compare("double") == 0) {
    return std::make_unique<DensityOCLMultiPlatform::OperationDensityOCLMultiPlatform<double>>(
        gridpoints, gridsize, dimension, manager, parameters, lambda);
  } else {
    std::stringstream errorString;
    errorString << "Error creating operation\"OperationDensityOCLMultiPlatform\": "
                << " invalid value for parameter \"INTERNAL_PRECISION\"";
    throw base::factory_exception(errorString.str().c_str());
  }
  return nullptr;
}
std::unique_ptr<DensityOCLMultiPlatform::OperationDensity> createDensityOCLMultiPlatformConfigured(
    const std::vector<int> &gridpoints, size_t gridsize, size_t dimension, double lambda,
    std::shared_ptr<base::OCLOperationConfiguration> parameters) {
  DensityOCLMultiPlatform::OperationDensity::load_default_parameters(parameters);
  auto manager =
      std::make_shared<base::OCLManagerMultiPlatform>(parameters);

  if ((*parameters)["INTERNAL_PRECISION"].get().compare("float") == 0) {
    return std::make_unique<DensityOCLMultiPlatform::OperationDensityOCLMultiPlatform<float>>(
        gridpoints, gridsize, dimension, manager, parameters, static_cast<float>(lambda));
  } else if ((*parameters)["INTERNAL_PRECISION"].get().compare("double") == 0) {
    return std::make_unique<DensityOCLMultiPlatform::OperationDensityOCLMultiPlatform<double>>(
        gridpoints, gridsize, dimension, manager, parameters, lambda);
  } else {
    std::stringstream errorString;
    errorString << "Error creating operation\"OperationDensityOCLMultiPlatform\": "
                << " invalid value for parameter \"INTERNAL_PRECISION\"";
    throw base::factory_exception(errorString.str().c_str());
  }
  return nullptr;
}
}  // namespace datadriven
}  // namespace sgpp
