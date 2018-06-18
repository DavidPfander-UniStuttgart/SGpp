// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#include <sgpp/base/datatypes/DataVector.hpp>
#include <sgpp/base/grid/Grid.hpp>
#include <sgpp/base/grid/GridStorage.hpp>
#include <sgpp/base/grid/generation/GridGenerator.hpp>
#include <sgpp/base/opencl/OCLOperationConfiguration.hpp>
#include <sgpp/datadriven/operation/hash/OperationCreateGraphOCL/OpFactory.hpp>
#include <sgpp/datadriven/operation/hash/OperationDensityOCLMultiPlatform/OpFactory.hpp>
#include <sgpp/datadriven/operation/hash/OperationPruneGraphOCL/OpFactory.hpp>
#include <sgpp/globaldef.hpp>
#include <sgpp/solver/sle/ConjugateGradients.hpp>

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "sgpp/datadriven/tools/ARFFTools.hpp"

int main() {
  size_t dimension = 10;
  size_t level = 5;  // level 6 for testing
  size_t k = 12;
  double lambda = 0.001;
  double threshold = 0.0;
  std::string filename = "datasets/friedman/friedman2_4d_small.arff";

  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  start = std::chrono::system_clock::now();

  std::cout << "Loading file: " << filename << std::endl;
  sgpp::datadriven::Dataset data = sgpp::datadriven::ARFFTools::readARFF(filename);
  sgpp::base::DataMatrix& dataset = data.getData();
  dimension = dataset.getNcols();
  std::cout << "Loaded " << dataset.getNcols() << " dimensional dataset with " << dataset.getNrows()
            << " datapoints." << std::endl;

  // Create Grid
  auto grid = std::unique_ptr<sgpp::base::Grid>(sgpp::base::Grid::createLinearGrid(dimension));
  sgpp::base::GridGenerator& gridGen = grid->getGenerator();
  gridGen.regular(level);
  size_t gridsize = grid->getStorage().getSize();
  std::cerr << "Grid created! Number of grid points:     " << gridsize << std::endl;

  sgpp::base::DataVector alpha(gridsize);
  sgpp::base::DataVector result(gridsize);
  alpha.setAll(1.0);

  auto solver = std::make_shared<sgpp::solver::ConjugateGradients>(1000, 0.0001);
  auto operation_mult =
      std::unique_ptr<sgpp::datadriven::DensityOCLMultiPlatform::OperationDensity>(
          sgpp::datadriven::createDensityOCLMultiPlatformConfigured(*grid, dimension, lambda,
                                                                    "MyOCLConf.cfg"));

  // operation_mult->mult(alpha, result);

  std::cout << "Creating rhs" << std::endl;
  sgpp::base::DataVector b(gridsize);
  operation_mult->generateb(dataset, b);

  // for (size_t i = 0; i < 300; i++)
  //   std::cout << b[i] << " ";
  // std::cout << std::endl;
  // std::ofstream out_rhs("rhs_erg_dim2_depth11.txt");
  // out_rhs.precision(17);
  // for (size_t i = 0; i < gridsize; ++i) {
  //   out_rhs << b[i] << " ";
  // }
  // out_rhs.close();

  std::cout << "Creating alpha" << std::endl;
  solver->solve(*operation_mult, alpha, b, false, true);
  // scale alphas to [0...1] -> why?
  double max = alpha.max();
  double min = alpha.min();
  for (size_t i = 0; i < gridsize; i++) alpha[i] = alpha[i] * 1.0 / (max - min);

  std::ofstream out_alpha("alpha_erg_dim2_depth11.txt");
  out_alpha.precision(17);
  for (size_t i = 0; i < gridsize; ++i) {
    out_alpha << alpha[i] << " ";
  }
  out_alpha.close();
  std::cout << "Starting graph creation..." << std::endl;
  auto operation_graph =
      std::unique_ptr<sgpp::datadriven::DensityOCLMultiPlatform::OperationCreateGraphOCL>(
          sgpp::datadriven::createNearestNeighborGraphConfigured(dataset, k, dimension,
                                                                 "MyOCLConf.cfg"));
  std::vector<int> graph(dataset.getNrows() * k);
  operation_graph->create_graph(graph);

  std::ofstream out("graph_erg_dim2_depth11.txt");
  // for (size_t i = 0; i < dataset.getNrows(); ++i) {
  //   for (size_t j = 0; j < k; ++j) {
  //     out << graph[i * k + j] << " ";
  //   }
  //   out << std::endl;
  // }
  // out.close();

  std::cout << "Starting graph pruning" << std::endl;
  sgpp::datadriven::DensityOCLMultiPlatform::OperationPruneGraphOCL* operation_prune =
      sgpp::datadriven::pruneNearestNeighborGraphConfigured(*grid, dimension, alpha, dataset,
                                                            threshold, k, "MyOCLConf.cfg");
  operation_prune->prune_graph(graph);

  // out.open("graph_pruned_erg_dim2_depth11.txt");
  // for (size_t i = 0; i < dataset.getNrows(); ++i) {
  //   for (size_t j = 0; j < k; ++j) {
  //     out << graph[i * k + j] << " ";
  //   }
  //   out << std::endl;
  // }
  // out.close();
  std::vector<int> node_cluster_map;
  sgpp::datadriven::DensityOCLMultiPlatform::OperationCreateGraphOCL::neighborhood_list_t clusters;
  sgpp::datadriven::DensityOCLMultiPlatform::OperationCreateGraphOCL::find_clusters(
      graph, k, node_cluster_map, clusters);
  out.open("cluster_erg.txt");
  for (size_t datapoint : node_cluster_map) {
    out << datapoint << " ";
  }
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "finished computation at " << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s" << std::endl;
}
