// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org
#include <unistd.h>
#include <chrono>
#include <iostream>
#include <sgpp/base/datatypes/DataMatrix.hpp>
#include <sgpp/base/datatypes/DataVector.hpp>
#include <sgpp/base/grid/Grid.hpp>
#include <sgpp/datadriven/operation/hash/OperationMPI/MPIEnviroment.hpp>
#include <sgpp/globaldef.hpp>
#include <sgpp/solver/sle/ConjugateGradients.hpp>
#include <string>
#include <vector>
#include "sgpp/datadriven/operation/hash/OperationMPI/OperationDensityMultMPI.hpp"
#include "sgpp/datadriven/operation/hash/OperationMPI/OperationDensityRhsMPI.hpp"
#include "sgpp/datadriven/operation/hash/OperationMPI/OperationPrunedGraphCreationMPI.hpp"
#include "sgpp/datadriven/tools/ARFFTools.hpp"

// arg 1: MPI Config File
// arg 2: Dataset
// arg 3: Gridlevel
int main(int argc, char *argv[]) {
  // Init MPI enviroment - always has to be done first - capture slaves
  sgpp::datadriven::clusteringmpi::MPIEnviroment::init(argc, argv, true);

  // Measure times
  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  start = std::chrono::system_clock::now();

  if (argc != 4) {
    std::cout << "Wrong arguments - consult source file for more informations!" << std::endl;
    sgpp::datadriven::clusteringmpi::MPIEnviroment::release();
    return 0;
  }

  // Loading dataset
  std::string filename = argv[2];
  std::cout << "Loading file: " << filename << std::endl;
  sgpp::datadriven::Dataset data = sgpp::datadriven::ARFFTools::readARFF(filename);
  sgpp::base::DataMatrix &dataset = data.getData();
  size_t dim = data.getDimension();
  size_t level = std::stoi(argv[3]);

  sgpp::base::OperationConfiguration network_conf(argv[1]);
  sgpp::datadriven::clusteringmpi::MPIEnviroment::connect_nodes(network_conf);

  int rank = sgpp::datadriven::clusteringmpi::MPIEnviroment::get_node_rank();

  // Create Grid
  std::chrono::time_point<std::chrono::high_resolution_clock> grid_creation_start,
      grid_creation_end;

  sgpp::base::Grid *grid = sgpp::base::Grid::createLinearGrid(dim);
  sgpp::base::GridGenerator &gridGen = grid->getGenerator();
  gridGen.regular(level);
  size_t gridsize = grid->getStorage().getSize();
  sgpp::base::DataVector alpha(gridsize);
  sgpp::base::DataVector result(gridsize);
  alpha.setAll(1.0);
  std::cerr << "Grid created! Number of grid points:     " << gridsize << std::endl;
  std::cout << std::endl << std::endl;


  // Create right hand side vector
  std::chrono::time_point<std::chrono::high_resolution_clock> rhs_start, rhs_end;
  rhs_start = std::chrono::system_clock::now();
  std::cout << "Create right hand side of density equation: " << std::endl;
  std::cout << "-------------------------------------------- " << std::endl;
  sgpp::datadriven::clusteringmpi::OperationDensityRhsMPI rhs_op(*grid, dataset, "MyOCLConf.cfg");
  sgpp::base::DataVector rhs(gridsize);
  rhs_op.generate_b(rhs);
  rhs_end = std::chrono::system_clock::now();
  std::cout << std::endl << std::endl;

  // Solve for alpha vector via CG solver
  std::cout << "Solve for alpha: " << std::endl;
  std::cout << "--------------- " << std::endl;
  sgpp::datadriven::clusteringmpi::OperationDensityMultMPI mult_op(*grid, 0.001, "MyOCLConf.cfg");
  std::chrono::time_point<std::chrono::high_resolution_clock> solver_start, solver_end;
  solver_start = std::chrono::system_clock::now();
  alpha.setAll(1.0);
  sgpp::solver::ConjugateGradients solver(1000, 0.001);
  solver.solve(mult_op, alpha, rhs, false, true);
  solver_end = std::chrono::system_clock::now();
  double max = alpha.max();
  double min = alpha.min();
  for (size_t i = 0; i < gridsize; i++) alpha[i] = alpha[i] * 1.0 / (max - min);
  std::cout << std::endl << std::endl;

  // Create and prune knn graph
  std::cout << "Create and prune graph: " << std::endl;
  std::cout << "----------------------- " << std::endl;
  std::chrono::time_point<std::chrono::high_resolution_clock> create_knn_start, create_knn_end;
  create_knn_start = std::chrono::system_clock::now();
  sgpp::datadriven::clusteringmpi::OperationPrunedGraphCreationMPI graph_op(
      *grid, alpha, dataset, 5, 0.0, "MyOCLConf.cfg");
  std::vector<int> knn_graph;
  graph_op.create_graph(knn_graph);
  create_knn_end = std::chrono::system_clock::now();
  std::cout << std::endl << std::endl;

  std::cout << "Find clusters in pruned graph: " << std::endl;
  std::cout << "------------------------------ " << std::endl;
  std::chrono::time_point<std::chrono::high_resolution_clock> find_clusters_start,
      find_clusters_end;
  find_clusters_start = std::chrono::system_clock::now();
  std::vector<int> node_cluster_map;
  sgpp::datadriven::DensityOCLMultiPlatform::OperationCreateGraphOCL::neighborhood_list_t clusters;
  sgpp::datadriven::DensityOCLMultiPlatform::OperationCreateGraphOCL::find_clusters(
      knn_graph, 5, node_cluster_map, clusters);
  find_clusters_end = std::chrono::system_clock::now();
  // Output ergs
  std::cout << "detected clusters: " << clusters.size() << std::endl;
  std::ofstream out("cluster_erg.txt");
  for (size_t datapoint : node_cluster_map) {
    out << datapoint << std::endl;
  }
  std::cout << std::endl << std::endl;


  // Calc time
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);

  // Output timings
  std::cout << "Runtimes: " << std::endl;
  std::cout << "--------- " << std::endl;
  std::cout << "rhs creation duration: "
            << std::chrono::duration_cast<std::chrono::seconds>(rhs_end - rhs_start).count()
            << "s" << std::endl;
  std::cout << "solver duration: "
            << std::chrono::duration_cast<std::chrono::seconds>(solver_end - solver_start).count()
            << "s" << std::endl;
  std::cout << "create knn operation duration: "
            << std::chrono::duration_cast<std::chrono::seconds>(create_knn_end - create_knn_start)
                     .count()
            << "s" << std::endl;
  std::cout << "find clusters duration: "
            << std::chrono::duration_cast<std::chrono::seconds>(find_clusters_end -
                                                                find_clusters_start).count()
            << "s" << std::endl;
  std::cout << "finished computation at " << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";
  std::cout << std::endl << std::endl;

  // Cleanup MPI enviroment
  std::cout << "Finishing: " << std::endl;
  std::cout << "---------- " << std::endl;
  sgpp::datadriven::clusteringmpi::MPIEnviroment::release();
  return 0;
}
