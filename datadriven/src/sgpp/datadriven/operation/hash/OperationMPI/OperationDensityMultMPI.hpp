#pragma once

#include <chrono>
#include "DensityWorker.hpp"
#include "sgpp/base/operation/hash/OperationMatrix.hpp"

namespace sgpp {
namespace datadriven {
namespace clusteringmpi {

class OperationDensityMultMPI : public base::OperationMatrix, public DensityWorker {
 private:
  bool verbose_timing;
 public:
  OperationDensityMultMPI(base::Grid &grid, double lambda, std::string ocl_conf_filename, bool verbose_timing)
      : MPIWorkerBase("DensityMultiplicationWorker"),
        DensityWorker(grid, lambda, ocl_conf_filename), verbose_timing(verbose_timing) {}
  virtual ~OperationDensityMultMPI() {}
  virtual void mult(base::DataVector &alphavector, base::DataVector &result) {
    start_sub_workers();
    double *alpha_ptr = alphavector.getPointer();
    this->alpha = std::vector<double>(alpha_ptr, alpha_ptr+alphavector.size());
    send_alpha();
    long datainfo[2];
    datainfo[0] = 0;
    datainfo[1] = static_cast<int>(result.getSize());

    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    start = std::chrono::system_clock::now();

    divide_workpackages(datainfo, result.getPointer());

    end = std::chrono::system_clock::now();
    if (verbose_timing) {
      std::cout << "MPI muliplication duration: "
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() /
    1000000.0 << std::endl;
    }

    long exitmessage[2] = {-2, -2};
    for (int dest = 1; dest < MPIEnviroment::get_sub_worker_count() + 1; dest++)
      MPI_Send(exitmessage, 2, MPI_LONG, dest, 1, sub_worker_comm);
  }
};

}  // namespace clusteringmpi
}  // namespace datadriven
}  // namespace sgpp
