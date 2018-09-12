// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org
#pragma once

#include <mpi.h>

#include <sgpp/datadriven/operation/hash/OperationDensityOCLMultiPlatform/OpFactory.hpp>
#include "MPIWorkerGridBase.hpp"
#include "MPIWorkerPackageBase.hpp"

#include <exception>
#include <sstream>
#include <string>

namespace sgpp {
namespace datadriven {
namespace clusteringmpi {
class DensityWorker : public MPIWorkerGridBase, public MPIWorkerPackageBase<double> {
 protected:
  double lambda;
  std::unique_ptr<DensityOCLMultiPlatform::OperationDensity> op;
  std::vector<double> alpha;
  size_t oldgridsize;

  void receive_and_send_initial_data(void) {
    receive_alpha();
    if (opencl_node) op->initialize_alpha(alpha);
    send_alpha();
  }
  void begin_opencl_operation(long *workpackage) {
    op->start_partial_mult(workpackage[0], workpackage[1]);
  }
  void finalize_opencl_operation(double *result_buffer, long *workpackage) {
    op->finish_partial_mult(result_buffer, workpackage[0], workpackage[1]);
  }

 public:
  DensityWorker()
      : MPIWorkerBase("DensityMultiplicationWorker"),
        MPIWorkerGridBase("DensityMultiplicationWorker"),
        MPIWorkerPackageBase("DensityMultiplicationWorker", 1) {
    alpha.reserve(complete_gridsize / (2 * grid_dimensions));
    // Receive lambda
    MPI_Bcast(&lambda, 1, MPI_DOUBLE, 0,
              MPIEnviroment::get_input_communicator());

    // Send lambda
    if (MPIEnviroment::get_sub_worker_count() > 0) {
      MPI_Bcast(&lambda, 1, MPI_DOUBLE, 0,
                MPIEnviroment::get_communicator());
    }

    // Create opencl operation
    if (opencl_node) {
      op = createDensityOCLMultiPlatformConfigured(
          gridpoints, complete_gridsize / (2 * grid_dimensions), grid_dimensions, lambda,
          parameters);
    }
  }
  DensityWorker(base::Grid &grid, double lambda)
      : MPIWorkerBase("DensityMultiplicationWorker"),
        MPIWorkerGridBase("DensityMultiplicationWorker", grid),
        MPIWorkerPackageBase("DensityMultiplicationWorker", 1) {
    alpha.reserve(complete_gridsize / (2 * grid_dimensions));
    // Send lambda to slaves
    if (MPIEnviroment::get_sub_worker_count() > 0) {
      MPI_Bcast(&lambda, 1, MPI_DOUBLE, 0,
                MPIEnviroment::get_communicator());
    }
  }
  DensityWorker(base::Grid &grid, double lambda, std::string ocl_conf_filename)
      : MPIWorkerBase("DensityMultiplicationWorker"),
        MPIWorkerGridBase("DensityMultiplicationWorker", grid),
        MPIWorkerPackageBase("DensityMultiplicationWorker", 1, ocl_conf_filename) {
    alpha.reserve(complete_gridsize / (2 * grid_dimensions));
    // Send lambda to slaves
    if (MPIEnviroment::get_sub_worker_count() > 0) {
      MPI_Bcast(&lambda, 1, MPI_DOUBLE, 0,
                MPIEnviroment::get_communicator());
    }
  }
  virtual ~DensityWorker(void) {
  }

 protected:
  void send_alpha(void) {
    // Send alpha vector
    if (MPIEnviroment::get_sub_worker_count() > 0) {
      MPI_Bcast(alpha.data(), complete_gridsize / (2 * grid_dimensions), MPI_DOUBLE, 0,
                MPIEnviroment::get_communicator());
    }
  }
  void receive_alpha(void) {
    // Receive alpha vector
    MPI_Bcast(alpha.data(), complete_gridsize / (2 * grid_dimensions), MPI_DOUBLE, 0,
              MPIEnviroment::get_input_communicator());
  }
};

}  // namespace clusteringmpi
}  // namespace datadriven
}  // namespace sgpp
