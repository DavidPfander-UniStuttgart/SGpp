// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#include <boost/program_options.hpp>

#include <chrono>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "sgpp/base/datatypes/DataVector.hpp"
#include "sgpp/base/grid/Grid.hpp"
#include "sgpp/base/grid/GridStorage.hpp"
#include "sgpp/base/grid/generation/GridGenerator.hpp"
#include "sgpp/base/grid/generation/functors/SurplusCoarseningFunctor.hpp"
#include "sgpp/base/grid/generation/functors/SurplusRefinementFunctor.hpp"
#include "sgpp/base/opencl/OCLOperationConfiguration.hpp"
#include "sgpp/datadriven/DatadrivenOpFactory.hpp"
#include "sgpp/datadriven/operation/hash/OperationCreateGraphOCL/OpFactory.hpp"
#include "sgpp/datadriven/operation/hash/OperationDensityMultiplicationAVX/OperationDensityMultiplicationAVX.hpp"
#include "sgpp/datadriven/operation/hash/OperationDensityOCLMultiPlatform/OpFactory.hpp"
#include "sgpp/datadriven/operation/hash/OperationPruneGraphOCL/OpFactory.hpp"
#include "sgpp/datadriven/tools/ARFFTools.hpp"
#include "sgpp/globaldef.hpp"
#include "sgpp/solver/sle/ConjugateGradients.hpp"

using namespace sgpp;

int main(int argc, char **argv) {
  std::string datasetFileName = "datasets/DR5/DR5_nowarnings_less05_train_small.arff";
  // std::string datasetFileName = "datasets/friedman/friedman1_10d_150000.arff";
  // std::string datasetFileName = "datasets/friedman/friedman1_10d_small.arff";
  size_t level = 3;
  double lambda = 1E-2;
  std::string configFileName = "config_ocl_float_i76700k_valgrind.cfg";
  // std::string configFileName = "config_ocl_float_i76700k.cfg";
  uint64_t k = 6;
  double threshold = 1E-2;

  // read dataset
  std::cout << "reading dataset...";
  datadriven::Dataset dataset = datadriven::ARFFTools::readARFF(datasetFileName);
  std::cout << "done" << std::endl;

  size_t dimension = dataset.getDimension();
  std::cout << "dimension: " << dimension << std::endl;

  base::DataMatrix &trainingData = dataset.getData();
  std::cout << "data points: " << trainingData.getNrows() << std::endl;

  // create grid
  std::unique_ptr<base::Grid> grid(base::Grid::createLinearGrid(dimension));
  base::GridGenerator &grid_generator = grid->getGenerator();
  grid_generator.regular(level);
  std::cout << "Initial grid created! Number of grid points: " << grid->getSize() << std::endl;

  // create solver
  auto solver = std::make_unique<solver::ConjugateGradients>(1000, 0.0001);

  base::DataVector alpha(grid->getSize());
  alpha.setAll(0.0);
  base::DataVector b(grid->getSize(), 0.0);

  // create density calculator
  {
    std::unique_ptr<datadriven::DensityOCLMultiPlatform::OperationDensity> operation_mult(
        datadriven::createDensityOCLMultiPlatformConfigured(*grid, dimension, lambda,
                                                            configFileName));

    std::cout << "Calculating right-hand side" << std::endl;

    operation_mult->generateb(trainingData, b);

    double last_duration_generate_b = operation_mult->getLastDurationB();
    std::cout << "last_duration_generate_b: " << last_duration_generate_b << std::endl;
    double ops_generate_b = static_cast<double>(grid->getSize()) *
                            static_cast<double>(trainingData.getNrows()) *
                            (6 * static_cast<double>(dimension) + 1) * 1E-9;
    std::cout << "ops_generate_b: " << ops_generate_b << std::endl;
    double flops_generate_b = ops_generate_b / last_duration_generate_b;
    std::cout << "flops_generate_b: " << flops_generate_b << std::endl;

    std::cout << "Solving density SLE" << std::endl;
    solver->solve(*operation_mult, alpha, b, false, true);

    double acc_duration_density = operation_mult->getAccDurationDensityMult();
    std::cout << "acc_duration_density: " << acc_duration_density << std::endl;

    size_t iterations = solver->getNumberIterations();
    double act_it = static_cast<double>(iterations + 1 + (iterations / 50));
    std::cout << "act_it: " << act_it << std::endl;
    // TODO: CONTINUE!!! test solver iterations retrieved correctly
    double ops_density = std::pow(static_cast<double>(grid->getSize()), 2.0) * act_it *
                         (14.0 * static_cast<double>(dimension) + 2.0) * 1E-9;
    std::cout << "ops_density: " << ops_density << " GOps" << std::endl;
    double flops_density = ops_density / acc_duration_density;
    std::cout << "flops_density: " << flops_density << " GFLOPS" << std::endl;
  }

  // scale alphas to [0, 1] -> smaller function values, use?
  double max = alpha.max();
  double min = alpha.min();
  for (size_t i = 0; i < grid->getSize(); i++) {
    alpha[i] = alpha[i] * 1.0 / (max - min);
  }

  std::vector<int> graph(trainingData.getNrows() * k, -1);
  {
    std::cout << "Starting graph creation..." << std::endl;
    auto operation_graph =
        std::unique_ptr<sgpp::datadriven::DensityOCLMultiPlatform::OperationCreateGraphOCL>(
            sgpp::datadriven::createNearestNeighborGraphConfigured(trainingData, k, dimension,
                                                                   configFileName));

    operation_graph->create_graph(graph);

    double acc_duration_create_graph = operation_graph->getAccDuration();
    std::cout << "acc_duration_create_graph: " << acc_duration_create_graph << std::endl;

    double ops_create_graph = std::pow(static_cast<double>(trainingData.getNrows()), 2.0) * 4.0 *
                              static_cast<double>(dimension) * 1E-9;
    std::cout << "ops_create_graph: " << ops_create_graph << " GOps" << std::endl;
    double flops_create_graph = ops_create_graph / acc_duration_create_graph;
    std::cout << "flops_create_graph: " << flops_create_graph << " GFLOPS" << std::endl;
  }

  {
    std::cout << "Pruning graph..." << std::endl;

    std::unique_ptr<sgpp::datadriven::DensityOCLMultiPlatform::OperationPruneGraphOCL>
        operation_prune(sgpp::datadriven::pruneNearestNeighborGraphConfigured(
            *grid, dimension, alpha, trainingData, threshold, k, configFileName));
    operation_prune->prune_graph(graph);

    double last_duration_prune_graph = operation_prune->getLastDuration();
    std::cout << "last_duration_prune_graph: " << last_duration_prune_graph << std::endl;

    // middlepoint between node and neighbor ops
    double ops_prune_graph = static_cast<double>(trainingData.getNrows()) *
                             static_cast<double>(grid->getSize()) * static_cast<double>(k + 1) *
                             (6.0 * static_cast<double>(dimension) + 2) * 1E-9;
    std::cout << "ops_prune_graph: " << ops_prune_graph << " GOps" << std::endl;
    double flops_prune_graph = ops_prune_graph / last_duration_prune_graph;
    std::cout << "flops_prune_graph: " << flops_prune_graph << " GFLOPS" << std::endl;

    std::ofstream pruned_graph_output("prune_graph_kernel.data");
    for (size_t i = 0; i < graph.size() / dimension; i++) {
      for (size_t cur_k = 0; cur_k < k; cur_k += 1) {
        if (cur_k > 0) {
          pruned_graph_output << ", ";
        }
        pruned_graph_output << graph[i * k + cur_k];
      }
      pruned_graph_output << std::endl;
    }
  }

  std::cout << std::endl << "all done!" << std::endl;
}
