#pragma once

#include "DensityRhsWorker.hpp"

namespace sgpp {
namespace datadriven {
namespace clusteringmpi {

class OperationDensityRhsMPI : public DensityRhsWorker {
 public:
  OperationDensityRhsMPI(base::Grid &grid, std::string dataset_filename,
                         std::string ocl_config_file)
      : MPIWorkerBase("DensityRHSWorker"), DensityRhsWorker(grid, dataset_filename, ocl_config_file) {}
  virtual ~OperationDensityRhsMPI() {
  }
  virtual void generate_b(base::DataVector &b) {
    DensityRhsWorker::MPIWorkerGridBase::start_sub_workers();
    long datainfo[2];
    datainfo[0] = 0;
    datainfo[1] = static_cast<int>(b.getSize());
    divide_workpackages(datainfo, b.getPointer());
    long exitmessage[2] = {-2, -2};
    for (int dest = 1; dest < MPIEnviroment::get_sub_worker_count() + 1; dest++)
      MPI_Send(exitmessage, 2, MPI_LONG, dest, 1, sub_worker_comm);
  }
};

}  // namespace clusteringmpi
}  // namespace datadriven
}  // namespace sgpp
