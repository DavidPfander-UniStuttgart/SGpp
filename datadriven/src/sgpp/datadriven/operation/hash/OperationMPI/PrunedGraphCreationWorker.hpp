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
  double *alpha;
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
    int buffer_size = 0;
    MPI_Status stat;
    MPI_Probe(0, 1, master_worker_comm, &stat);
    MPI_Get_count(&stat, MPI_DOUBLE, &buffer_size);
    if (buffer_size != gridsize) {
      std::stringstream errorString;
      errorString << "Error: Gridsize " << gridsize << " and the size of the alpha vector "
                  << buffer_size << " should match!" << std::endl;
      throw std::logic_error(errorString.str());
    }
    alpha = new double[gridsize];

    MPI_Recv(alpha, gridsize, MPI_DOUBLE, stat.MPI_SOURCE, stat.MPI_TAG, master_worker_comm, &stat);

    // Send alpha vector
    for (int dest = 1; dest < MPIEnviroment::get_sub_worker_count() + 1; dest++)
      MPI_Send(alpha, static_cast<int>(complete_gridsize / (2 * grid_dimensions)), MPI_DOUBLE, dest,
               1, sub_worker_comm);

    // Receive treshold
    MPI_Recv(&treshold, 1, MPI_DOUBLE, stat.MPI_SOURCE, stat.MPI_TAG, master_worker_comm, &stat);

    // Send treshold
    for (int dest = 1; dest < MPIEnviroment::get_sub_worker_count() + 1; dest++)
      MPI_Send(&treshold, static_cast<int>(1), MPI_DOUBLE, dest, 1, sub_worker_comm);
    // Create opencl operation
    if (opencl_node) {
      op = createNearestNeighborGraphConfigured(dataset, dataset_size, k, dimensions,
                                                parameters);  // TODO opencl_platform, opencl_device
      op_prune = pruneNearestNeighborGraphConfigured(
          gridpoints.data(), complete_gridsize / (2 * grid_dimensions), grid_dimensions, alpha,
          data_matrix, treshold, k, parameters);  // , opencl_platform, opencl_device
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
    for (int dest = 1; dest < MPIEnviroment::get_sub_worker_count() + 1; dest++)
      MPI_Send(alpha.getPointer(), static_cast<int>(complete_gridsize / (2 * grid_dimensions)),
               MPI_DOUBLE, dest, 1, sub_worker_comm);
    // Send treshold
    for (int dest = 1; dest < MPIEnviroment::get_sub_worker_count() + 1; dest++)
      MPI_Send(&treshold, static_cast<int>(1), MPI_DOUBLE, dest, 1, sub_worker_comm);
  }
  virtual ~PrunedGraphCreationWorker(void) {
    if (delete_alpha) {
      delete[] alpha;
    }
    if (opencl_node) {
      delete op;
      delete op_prune;
    }
  }
};

}  // namespace clusteringmpi
}  // namespace datadriven
}  // namespace sgpp
