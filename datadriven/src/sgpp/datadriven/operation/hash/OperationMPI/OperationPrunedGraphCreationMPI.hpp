#pragma once

#include "PrunedGraphCreationWorker.hpp"

namespace sgpp {
namespace datadriven {
namespace clusteringmpi {

class OperationPrunedGraphCreationMPI : public PrunedGraphCreationWorker {
 public:
  OperationPrunedGraphCreationMPI(base::Grid &grid, base::DataVector &alpha, base::DataMatrix &data,
                                  int k, double treshold, std::string ocl_conf_filename)
      : MPIWorkerBase("PrunedGraphCreationWorker"),
        PrunedGraphCreationWorker(grid, alpha, data, k, treshold, ocl_conf_filename) {}
  virtual ~OperationPrunedGraphCreationMPI() {}
  virtual void create_graph(std::vector<int64_t> &result) {
    start_sub_workers();
    long datainfo[2];
    datainfo[0] = 0;
    datainfo[1] = dataset_size / dimensions;

    result.resize(dataset_size / dimensions * k);
    divide_workpackages(datainfo, result.data());

    long exitmessage[2] = {-2, -2};
    for (int dest = 1; dest < MPIEnviroment::get_sub_worker_count() + 1; dest++)
      MPI_Send(exitmessage, 2, MPI_LONG, dest, 1, sub_worker_comm);
  }
};

}  // namespace clusteringmpi
}  // namespace datadriven
}  // namespace sgpp
