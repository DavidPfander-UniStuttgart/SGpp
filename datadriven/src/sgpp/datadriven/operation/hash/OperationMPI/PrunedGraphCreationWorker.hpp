// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org
#pragma once

#include <algorithm>
#include <sgpp/datadriven/operation/hash/OperationCreateGraphOCL/OpFactory.hpp>
#include <sgpp/datadriven/operation/hash/OperationCreateGraphOCL/OperationCreateGraphOCL.hpp>
#include <sgpp/datadriven/operation/hash/OperationDensityOCLMultiPlatform/OpFactory.hpp>
#include <sgpp/datadriven/operation/hash/OperationPruneGraphOCL/OpFactory.hpp>
#include <sgpp/datadriven/operation/hash/OperationPruneGraphOCL/OperationPruneGraphOCL.hpp>
#include <string>
#include <vector>

namespace sgpp {
namespace datadriven {
namespace clusteringmpi {
class PrunedGraphCreationWorker : public MPIWorkerGridBase,
                                  public MPIWorkerGraphBase,
                                  public MPIWorkerPackageBase<int> {
 private:
  bool delete_alpha;

 protected:
  double treshold;
  std::vector<double> alpha;
  DensityOCLMultiPlatform::OperationCreateGraphOCL *op;
  DensityOCLMultiPlatform::OperationPruneGraphOCL *op_prune;
  void receive_and_send_initial_data(void) {}
  void begin_opencl_operation(int *workpackage) {
    op->begin_graph_creation(workpackage[0], workpackage[1]);
  }
  void finalize_opencl_operation(int *result_buffer, int *workpackage) {
    std::vector<int> partial_graph(workpackage[1] * packagesize_multiplier);
    op->finalize_graph_creation(partial_graph, workpackage[0], workpackage[1]);
    op_prune->prune_graph(partial_graph, workpackage[0], workpackage[1]);
    std::copy(partial_graph.begin(), partial_graph.end(), result_buffer);
  }

 public:
  PrunedGraphCreationWorker()
      : MPIWorkerBase("PrunedGraphCreationWorker"),
        MPIWorkerGridBase("PrunedGraphCreationWorker"),
        MPIWorkerGraphBase("PrunedGraphCreationWorker"),
        MPIWorkerPackageBase("PrunedGraphCreationWorker", k),
        delete_alpha(true) {
    // Create datamatrix for operation creation
    base::DataMatrix data_matrix(dataset, dataset_size / dimensions, dimensions);
    // Receive alpha vector
    int gridsize = complete_gridsize / (2 * grid_dimensions);
    MPI_Status stat;
    alpha.reserve(complete_gridsize / (2 * grid_dimensions));

    // Receive alpha vector
    MPI_Bcast(alpha.data(), complete_gridsize / (2 * grid_dimensions), MPI_DOUBLE, 0,
              MPIEnviroment::get_input_communicator());

    // Send alpha vector
    if (MPIEnviroment::get_sub_worker_count() > 0) {
      MPI_Bcast(alpha.data(), complete_gridsize / (2 * grid_dimensions), MPI_DOUBLE, 0,
                MPIEnviroment::get_communicator());
    }

    // Receive treshold
    MPI_Recv(&treshold, 1, MPI_DOUBLE, 0, 1, master_worker_comm, &stat);

    // Send treshold
    for (int dest = 1; dest < MPIEnviroment::get_sub_worker_count() + 1; dest++)
      MPI_Send(&treshold, static_cast<int>(1), MPI_DOUBLE, dest, 1, sub_worker_comm);
    // Create opencl operation
    if (opencl_node) {
      op = createNearestNeighborGraphConfigured(dataset, dataset_size, k, dimensions,
                                                parameters);
      op_prune = pruneNearestNeighborGraphConfigured(
          gridpoints, complete_gridsize / (2 * grid_dimensions), grid_dimensions, alpha,
          data_matrix, treshold, k, parameters);
    }
  }
  PrunedGraphCreationWorker(base::Grid &grid, base::DataVector &alpha, std::string dataset_filename,
                            int k, double treshold, std::string ocl_conf_filename)
      : MPIWorkerBase("PrunedGraphCreationWorker"),
        MPIWorkerGridBase("PrunedGraphCreationWorker", grid),
        MPIWorkerGraphBase("PrunedGraphCreationWorker", dataset_filename, k),
        MPIWorkerPackageBase("PrunedGraphCreationWorker", k, ocl_conf_filename),
        delete_alpha(false) {
    // Send alpha vector
    MPI_Bcast(alpha.getPointer(), complete_gridsize / (2 * grid_dimensions), MPI_DOUBLE, 0,
              MPIEnviroment::get_communicator());
    // Send treshold
    for (int dest = 1; dest < MPIEnviroment::get_sub_worker_count() + 1; dest++)
      MPI_Send(&treshold, static_cast<int>(1), MPI_DOUBLE, dest, 1, sub_worker_comm);
  }
  virtual ~PrunedGraphCreationWorker(void) {
    if (opencl_node) {
      delete op;
      delete op_prune;
    }
  }
};

}  // namespace clusteringmpi
}  // namespace datadriven
}  // namespace sgpp
