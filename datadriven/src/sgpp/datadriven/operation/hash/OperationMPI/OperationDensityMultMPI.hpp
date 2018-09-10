#pragma once

#include "DensityWorker.hpp"
#include "sgpp/base/operation/hash/OperationMatrix.hpp"

namespace sgpp {
namespace datadriven {
namespace clusteringmpi {

class OperationDensityMultMPI : public base::OperationMatrix, public DensityWorker {
 public:
  OperationDensityMultMPI(base::Grid &grid, double lambda, std::string ocl_conf_filename)
      : MPIWorkerBase("DensityMultiplicationWorker"),
        DensityWorker(grid, lambda, ocl_conf_filename) {}
  virtual ~OperationDensityMultMPI() {}
  virtual void mult(base::DataVector &alphavector, base::DataVector &result) {
    start_sub_workers();
    double *alpha_ptr = alphavector.getPointer();
    this->alpha = std::vector<double>(alpha_ptr, alpha_ptr+alphavector.size());
    send_alpha();
    int datainfo[2];
    datainfo[0] = 0;
    datainfo[1] = static_cast<int>(result.getSize());
    divide_workpackages(datainfo, result.getPointer());
    int exitmessage[2] = {-2, -2};
    for (int dest = 1; dest < MPIEnviroment::get_sub_worker_count() + 1; dest++)
      MPI_Send(exitmessage, 2, MPI_INT, dest, 1, sub_worker_comm);
  }
};

}  // namespace clusteringmpi
}  // namespace datadriven
}  // namespace sgpp
