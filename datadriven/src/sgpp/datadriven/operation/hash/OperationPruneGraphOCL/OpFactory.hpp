// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#pragma once

#include <sgpp/globaldef.hpp>
#include <string>
#include "OperationPruneGraphOCLMultiPlatform.hpp"
namespace sgpp {
namespace datadriven {

/// Generates the graph pruning operation for a specific opencl device
sgpp::datadriven::DensityOCLMultiPlatform::OperationPruneGraphOCL *
pruneNearestNeighborGraphConfigured(base::Grid &grid, size_t dimensions, base::DataVector &alpha,
                                    base::DataMatrix &data, double threshold, size_t k,
                                    std::string opencl_conf);
/// Generates the graph pruning operation for a specific opencl device using a serialized grid
sgpp::datadriven::DensityOCLMultiPlatform::OperationPruneGraphOCL *
pruneNearestNeighborGraphConfigured(const std::vector<int> &gridpoints, size_t gridsize, size_t dimensions,
                                    double *alpha, base::DataMatrix &data, double threshold,
                                    size_t k, std::string opencl_conf);
DensityOCLMultiPlatform::OperationPruneGraphOCL *pruneNearestNeighborGraphConfigured(
    const std::vector<int> &gridpoints, size_t gridsize, size_t dimensions, double *alpha, base::DataMatrix &data,
    double threshold, size_t k, std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters);
}  // namespace datadriven
}  // namespace sgpp
