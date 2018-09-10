// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#ifndef CREATE_GRAPH_OPFACTOR_H
#define CREATE_GRAPH_OPFACTOR_H

#include <sgpp/globaldef.hpp>
#include <string>
#include "OperationCreateGraphOCLSingleDevice.hpp"
namespace sgpp {
namespace datadriven {

/// Generates the k nearest neighbors graph creation using a specific opencl device and a datamatrix
std::unique_ptr<DensityOCLMultiPlatform::OperationCreateGraphOCL> createNearestNeighborGraphConfigured(
    base::DataMatrix &dataset, size_t k, size_t dimensions, std::string opencl_conf);
/// Generates the k nearest neighbors graph creation using a specific opencl device and a double
/// vector
std::unique_ptr<DensityOCLMultiPlatform::OperationCreateGraphOCL> createNearestNeighborGraphConfigured(
    double *dataset, size_t dataset_size, size_t k, size_t dimensions, std::string opencl_conf);
std::unique_ptr<DensityOCLMultiPlatform::OperationCreateGraphOCL> createNearestNeighborGraphConfigured(
    double *dataset, size_t dataset_size, size_t k, size_t dimensions,
    std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters);
}  // namespace datadriven
}  // namespace sgpp

#endif /* CREATE_GRAPH_OPFACTOR_H */
