// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org
#ifndef OPERATIONGRAPHBASEMPI_H
#define OPERATIONGRAPHBASEMPI_H

#include <mpi.h>
#include <sgpp/base/datatypes/DataMatrix.hpp>
#include <sgpp/datadriven/operation/hash/OperationMPI/MPIEnviroment.hpp>
#include <sgpp/datadriven/operation/hash/OperationMPI/MPIWorkerBase.hpp>
#include <sgpp/globaldef.hpp>
// #include <sgpp/datadriven/tools/ARFFTools.hpp>
#include <string>

namespace sgpp {
namespace datadriven {
namespace clusteringmpi {
class MPIWorkerGraphBase : virtual public MPIWorkerBase {
 protected:
  double *dataset;
  int dataset_size;
  int k;
  int dimensions;

     MPIWorkerGraphBase(std::string operationName, sgpp::base::DataMatrix &data, int k)
      : MPIWorkerBase(operationName),
        dataset(data.getPointer()),
        dataset_size(static_cast<int>(data.getSize())),
        k(k),
        dimensions(static_cast<int>(data.getNcols())),
        delete_dataset(false) {
    send_dataset();
  }
  //     MPIWorkerGraphBase(std::string operationName, sgpp::base::DataMatrix &data, int k)
  //     : MPIWorkerBase(operationName),
  //       dataset(data.getPointer()),
  //       dataset_size(static_cast<int>(data.getSize())),
  //       k(k),
  //       dimensions(static_cast<int>(data.getNcols())),
  //       delete_dataset(false) {
  //   send_dataset();
  // }
  explicit MPIWorkerGraphBase(std::string operationName)
      : MPIWorkerBase(operationName), delete_dataset(true) {
    receive_dataset();
    send_dataset();
  }

  virtual ~MPIWorkerGraphBase(void) {
    if (delete_dataset) delete [] dataset;
  }

 private:
  bool delete_dataset;
  void send_dataset() {
    MPI_Bcast(&dataset_size, 1, MPI_INT, 0, MPIEnviroment::get_communicator());
    MPI_Bcast(dataset, dataset_size, MPI_DOUBLE, 0, MPIEnviroment::get_communicator());
    MPI_Bcast(&dimensions, 1, MPI_INT, 0, MPIEnviroment::get_communicator());
    MPI_Bcast(&k, 1, MPI_INT, 0, MPIEnviroment::get_communicator());
  }
  // void load_whole_dataset(std::string filename) {
  //   data = sgpp::datadriven::ARFFTools::readARFF(filename);
  //   sgpp::base::DataMatrix &datamatrix = data.getData();
  //   dimensions = datamatrix.getNcols();
  //   dataset_size = datamatrix.getSize();
  //   dataset = datamatrix.getPointer();
  // }
  // void load_distributed_dataset(std::string filename) {
  //   // 1. Requires ARFF Header
  //   // 2. Requieres Starting Point (in Bytes)
  //   // 3. Requires chunksize
  //   // 4. Requires overall size? MPI_File_get_Size
  //   // 3. Start Loading

  //   data = sgpp::datadriven::ARFFTools::readARFF(filename);
  //   sgpp::base::DataMatrix &datamatrix = data.getData();
  //   dimensions = datamatrix.getNcols();
  //   dataset_size = datamatrix.getSize();
  //   dataset = datamatrix.getPointer();
  // }
  void receive_dataset(void) {
    MPI_Status stat;
    // Receive dataset
    MPI_Bcast(&dataset_size, 1, MPI_INT, 0, MPIEnviroment::get_input_communicator());
    dataset = new double[dataset_size];
    MPI_Bcast(dataset, dataset_size, MPI_DOUBLE, 0, MPIEnviroment::get_input_communicator());
    MPI_Bcast(&dimensions, 1, MPI_INT, 0, MPIEnviroment::get_input_communicator());
    MPI_Bcast(&k, 1, MPI_INT, 0, MPIEnviroment::get_input_communicator());

  }
};

}  // namespace clusteringmpi
}  // namespace datadriven
}  // namespace sgpp
#endif /* OPERATIONGRAPHBASEMPI_H */
