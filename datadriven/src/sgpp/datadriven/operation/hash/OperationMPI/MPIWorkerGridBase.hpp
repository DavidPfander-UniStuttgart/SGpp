// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org
#pragma once

#include <mpi.h>

#include <sgpp/base/grid/Grid.hpp>
#include <sgpp/datadriven/operation/hash/OperationMPI/MPIEnviroment.hpp>
#include <sgpp/datadriven/operation/hash/OperationMPI/MPIWorkerBase.hpp>

#include <string>

namespace sgpp {
namespace datadriven {
namespace clusteringmpi {

class MPIWorkerGridBase : virtual public MPIWorkerBase {
 private:
  void receive_grid(void) {
    // Receive gridsize
    MPI_Bcast(&complete_gridsize, 1, MPI_INT, 0, MPIEnviroment::get_input_communicator());
    // Receive grid
    gridpoints.reserve(complete_gridsize);
    MPI_Bcast(gridpoints.data(), complete_gridsize, MPI_INT, 0, MPIEnviroment::get_input_communicator());
    // Receive grid dimension
    MPI_Bcast(&grid_dimensions, 1, MPI_INT, 0, MPIEnviroment::get_input_communicator());
  }
  void send_grid(void) {
    // Send gridsize
    MPI_Bcast(&complete_gridsize, 1, MPI_INT, 0, MPIEnviroment::get_communicator());
    // Send grid
    MPI_Bcast(gridpoints.data(), complete_gridsize, MPI_INT, 0, MPIEnviroment::get_communicator());
    // Send grid dimension
    MPI_Bcast(&grid_dimensions, 1, MPI_INT, 0, MPIEnviroment::get_communicator());
  }

 protected:
  int grid_dimensions;
  int complete_gridsize;
  int gridsize;
  std::vector<int> gridpoints;
  explicit MPIWorkerGridBase(std::string operationName) :
      MPIWorkerBase(operationName) {
    receive_grid();
    send_grid();
  }
  MPIWorkerGridBase(std::string operationName, base::Grid &grid) : MPIWorkerBase(operationName) {
    // Store grid in integer array
    sgpp::base::GridStorage &gridStorage = grid.getStorage();
    gridsize = static_cast<int>(gridStorage.getSize());
    int dimensions = static_cast<int>(gridStorage.getDimension());
    gridpoints.reserve(gridsize * 2 * dimensions);
    size_t pointscount = 0;
    for (int i = 0; i < gridsize; i++) {
      sgpp::base::HashGridPoint &point = gridStorage.getPoint(i);
      pointscount++;
      for (int d = 0; d < dimensions; d++) {
        gridpoints[i * 2 * dimensions + 2 * d] = point.getIndex(d);
        gridpoints[i * 2 * dimensions + 2 * d + 1] = point.getLevel(d);
      }
    }

    complete_gridsize = gridsize * 2 * dimensions;
    grid_dimensions = dimensions;
    send_grid();
  }

 public:
  virtual ~MPIWorkerGridBase() {}
};

}  // namespace clusteringmpi
}  // namespace datadriven
}  // namespace sgpp
