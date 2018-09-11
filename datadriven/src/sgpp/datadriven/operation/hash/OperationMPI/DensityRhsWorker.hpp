// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org
#pragma once

#include <sgpp/datadriven/operation/hash/OperationDensityOCLMultiPlatform/OpFactory.hpp>
#include <sgpp/datadriven/operation/hash/OperationMPI/MPIWorkerGraphBase.hpp>
#include <sgpp/datadriven/operation/hash/OperationMPI/MPIWorkerGridBase.hpp>
#include <sgpp/datadriven/operation/hash/OperationMPI/MPIWorkerPackageBase.hpp>
#include <string>

namespace sgpp {
namespace datadriven {
namespace clusteringmpi {
class DensityRhsWorker : public MPIWorkerGridBase,
                         public MPIWorkerGraphBase,
                         public MPIWorkerPackageBase<double> {
 protected:
  double lambda;
  std::shared_ptr<DensityOCLMultiPlatform::OperationDensity> op;
  base::DataMatrix *data_matrix;

  void receive_and_send_initial_data(void) {
    if (data_matrix != NULL) delete data_matrix;
    data_matrix = new base::DataMatrix(dataset, dataset_size / dimensions, dimensions);
    if (opencl_node) op->initialize_dataset(*data_matrix);
  }
  void begin_opencl_operation(long *workpackage) {
    op->start_rhs_generation(workpackage[0], workpackage[1]);
  }
  void finalize_opencl_operation(double *result_buffer, long *workpackage) {
    base::DataVector partial_rhs(workpackage[1]);
    op->finalize_rhs_generation(partial_rhs, workpackage[0], workpackage[1]);
    for (int i = 0; i < workpackage[1]; ++i) {
      result_buffer[i] = partial_rhs[i];
    }
  }

 public:
  DensityRhsWorker()
      : MPIWorkerBase("DensityRHSWorker"),
        MPIWorkerGridBase("DensityRHSWorker"),
        MPIWorkerGraphBase("DensityRHSWorker"),
        MPIWorkerPackageBase("DensityMultiplicationWorker", 1) {
    // Create opencl operation
    if (opencl_node) {
      op = std::shared_ptr<DensityOCLMultiPlatform::OperationDensity>(
          createDensityOCLMultiPlatformConfigured(
              gridpoints, complete_gridsize / (2 * grid_dimensions), grid_dimensions, 0.0,
              parameters));  // TODO: opencl_platform, opencl_device
    }
    data_matrix = NULL;
  }
  DensityRhsWorker(base::Grid &grid, std::string dataset_filename, std::string ocl_config_file)
      : MPIWorkerBase("DensityRHSWorker"),
        MPIWorkerGridBase("DensityRHSWorker", grid),
        MPIWorkerGraphBase("DensityRHSWorker", dataset_filename, 0),
        MPIWorkerPackageBase("DensityMultiplicationWorker", 1, ocl_config_file) {
    data_matrix = NULL;
  }
};
}  // namespace clusteringmpi
}  // namespace datadriven
}  // namespace sgpp
