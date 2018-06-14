// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org
#pragma once
#include <sgpp/datadriven/operation/hash/OperationMPI/DensityRhsWorker.hpp>
#include <sgpp/datadriven/operation/hash/OperationMPI/DensityWorker.hpp>
#include <sgpp/datadriven/operation/hash/OperationMPI/GraphCreationWorker.hpp>
#include <sgpp/datadriven/operation/hash/OperationMPI/PrunedGraphCreationWorker.hpp>
namespace sgpp {
namespace datadriven {
namespace clusteringmpi {

MPIWorkerBase* create_mpi_operation(int masternode, base::OperationConfiguration conf,
                                    char* classname) {
  if (std::strcmp(classname, "OPDummy") == 0) {
    return new WorkerDummy(classname);
  }
  if (std::strcmp(classname, "DensityMultiplicationWorker") == 0) {
    return new DensityWorker();
  }
  if (std::strcmp(classname, "DensityRHSWorker") == 0) {
    return new DensityRhsWorker();
  }
  if (std::strcmp(classname, "GraphCreationWorker") == 0) {
    return new GraphCreationWorker();
  }
  if (std::strcmp(classname, "PrunedGraphCreationWorker") == 0) {
    return new PrunedGraphCreationWorker();
  }
  return NULL;
}

}  // namespace clusteringmpi
}  // namespace datadriven
}  // namespace sgpp
