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
#include <sgpp/datadriven/tools/ARFFTools.hpp>
#include <string>

namespace sgpp {
namespace datadriven {
namespace clusteringmpi {
class MPIWorkerGraphBase : virtual public MPIWorkerBase {
 protected:
  sgpp::datadriven::Dataset data;
  double *dataset;
  int dataset_size;
  int k;
  int dimensions;
  std::string dataset_filename;

  MPIWorkerGraphBase(std::string operationName, std::string filename, int k)
      : MPIWorkerBase(operationName), dataset_filename(filename),
        k(k) {
    load_dataset(dataset_filename);
    send_dataset();
  }
  MPIWorkerGraphBase(std::string filename, int k)
      : MPIWorkerBase(), dataset_filename(filename),
        k(k) {
    load_dataset(dataset_filename);
    send_dataset();
  }
  explicit MPIWorkerGraphBase(std::string operationName)
      : MPIWorkerBase(operationName) {
    receive_dataset();
    load_dataset(dataset_filename);
    send_dataset();
  }

  virtual ~MPIWorkerGraphBase(void) {
  }

 private:
  void send_dataset() {
    // Sending filename to slaves
    for (int i = 1; i < MPIEnviroment::get_sub_worker_count() + 1; i++) {
      MPI_Send(dataset_filename.c_str(), dataset_filename.size() + 1, MPI_CHAR, i, 1,
               MPIEnviroment::get_communicator());
    }
    // Sending k to slaves
    for (int i = 1; i < MPIEnviroment::get_sub_worker_count() + 1; i++) {
      MPI_Send(&k, 1, MPI_INT, i, 1, MPIEnviroment::get_communicator());
    }
  }
  void load_dataset(std::string filename) {
    data = sgpp::datadriven::ARFFTools::readARFF(filename);
    sgpp::base::DataMatrix &datamatrix = data.getData();
    dimensions = datamatrix.getNcols();
    dataset_size = datamatrix.getSize();
    dataset = datamatrix.getPointer();
  }
  void receive_dataset(void) {
    MPI_Status stat;
    // Receive dataset filename
    int filename_size = 0;
    MPI_Probe(0, 1, MPIEnviroment::get_input_communicator(), &stat);
    MPI_Get_count(&stat, MPI_CHAR, &filename_size);
    char *filename_tmp = new char[filename_size];
    MPI_Recv(filename_tmp, filename_size, MPI_CHAR, stat.MPI_SOURCE, stat.MPI_TAG,
             MPIEnviroment::get_input_communicator(), &stat);
    dataset_filename = std::string(filename_tmp);
    delete [] filename_tmp;
    // Receive clustering parameters
    MPI_Probe(0, 1, MPIEnviroment::get_input_communicator(), &stat);
    MPI_Recv(&k, 1, MPI_INT, stat.MPI_SOURCE, stat.MPI_TAG, MPIEnviroment::get_input_communicator(),
             &stat);
  }
};

}  // namespace clusteringmpi
}  // namespace datadriven
}  // namespace sgpp
#endif /* OPERATIONGRAPHBASEMPI_H */
