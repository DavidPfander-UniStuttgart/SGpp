// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#include <boost/program_options.hpp>

#include <chrono>
#include <experimental/filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "sgpp/base/datatypes/DataVector.hpp"
#include "sgpp/base/grid/Grid.hpp"
#include "sgpp/base/grid/GridStorage.hpp"
#include "sgpp/base/grid/generation/GridGenerator.hpp"
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
  std::string datasetFileName;
  size_t eval_grid_level;
  size_t level;
  double lambda;
  std::string configFileName;
  uint64_t k;
  double threshold;

  boost::program_options::options_description description("Allowed options");

  description.add_options()("help", "display help")(
      "datasetFileName", boost::program_options::value<std::string>(&datasetFileName),
      "training data set as an arff file")(
      "eval_grid_level", boost::program_options::value<size_t>(&eval_grid_level)->default_value(2),
      "level for the evaluation of the sparse grid density function (for picture creation)")(
      "level", boost::program_options::value<size_t>(&level)->default_value(4),
      "level of the sparse grid used for density estimation")(
      "lambda", boost::program_options::value<double>(&lambda)->default_value(0.000001),
      "regularization for density estimation")(
      "config", boost::program_options::value<std::string>(&configFileName),
      "OpenCL and kernel configuration file")(
      "k", boost::program_options::value<uint64_t>(&k)->default_value(5),
      "specifies number of neighbors for kNN algorithm")(
      "threshold", boost::program_options::value<double>(&threshold)->default_value(0.0),
      "threshold for sparse grid function for removing edges");

  boost::program_options::variables_map variables_map;

  boost::program_options::parsed_options options = parse_command_line(argc, argv, description);
  boost::program_options::store(options, variables_map);
  boost::program_options::notify(variables_map);

  if (variables_map.count("help")) {
    std::cout << description << std::endl;
    return 0;
  }

  if (variables_map.count("datasetFileName") == 0) {
    std::cerr << "error: option \"datasetFileName\" not specified" << std::endl;
    return 1;
  } else {
    std::experimental::filesystem::path datasetFilePath(datasetFileName);
    if (!std::experimental::filesystem::exists(datasetFilePath)) {
      std::cerr << "error: dataset file does not exist: " << datasetFileName << std::endl;
      return 1;
    }
    std::cout << "datasetFileName: " << datasetFileName << std::endl;
  }

  if (variables_map.count("eval_grid_level") == 0) {
    std::cerr << "error: option \"eval_grid_level\" not specified" << std::endl;
    return 1;
  } else {
    std::cout << "eval_grid_level: " << eval_grid_level << std::endl;
  }

  if (variables_map.count("level") == 0) {
    std::cerr << "error: option \"level\" not specified" << std::endl;
    return 1;
  } else {
    std::cout << "level: " << level << std::endl;
  }

  if (variables_map.count("lambda") == 0) {
    std::cerr << "error: option \"lambda\" not specified" << std::endl;
    return 1;
  } else {
    std::cout << "lambda: " << lambda << std::endl;
  }

  if (variables_map.count("config") == 0) {
    std::cerr << "error: option \"config\" not specified" << std::endl;
    return 1;
  } else {
    std::experimental::filesystem::path configFilePath(configFileName);
    if (!std::experimental::filesystem::exists(configFilePath)) {
      std::cerr << "error: config file does not exist: " << configFileName << std::endl;
      return 1;
    }

    std::cout << "OpenCL configuration file: " << configFileName << std::endl;
  }

  if (variables_map.count("k") == 0) {
    std::cerr << "error: option \"k\" not specified" << std::endl;
    return 1;
  } else {
    std::cout << "k: " << k << std::endl;
  }

  if (variables_map.count("threshold") == 0) {
    std::cerr << "error: option \"threshold\" not specified" << std::endl;
    return 1;
  } else {
    std::cout << "threshold: " << threshold << std::endl;
  }

  // read dataset
  std::cout << "reading dataset...";
  datadriven::Dataset dataset = datadriven::ARFFTools::readARFF(datasetFileName);
  std::cout << "done" << std::endl;
  size_t dimension = dataset.getDimension();
  std::cout << "dimension: " << dimension << std::endl;
  base::DataMatrix &trainingData = dataset.getData();

  // create grid
  base::Grid *grid = base::Grid::createLinearGrid(dimension);
  base::GridGenerator &gridGen = grid->getGenerator();
  gridGen.regular(level);
  size_t gridsize = grid->getStorage().getSize();
  std::cout << "Grid created! Number of grid points:     " << gridsize << std::endl;

  base::DataVector alpha(gridsize);
  base::DataVector result(gridsize);
  alpha.setAll(1.0);  // TODO: why 1.0?

  // create solver
  auto solver = std::make_unique<solver::ConjugateGradients>(1000, 0.0001);

  // create density calculator
  auto operation_mult = std::unique_ptr<datadriven::DensityOCLMultiPlatform::OperationDensity>(
      datadriven::createDensityOCLMultiPlatformConfigured(*grid, dimension, lambda,
                                                          configFileName));

  std::cout << "Calculating right-hand side" << std::endl;
  base::DataVector b(gridsize, 0.0);
  operation_mult->generateb(trainingData, b);
  // for (size_t i = 0; i < 300; i++) std::cout << b[i] << " ";
  // std::cout << std::endl;
  // std::ofstream out_rhs("rhs_erg_dim2_depth11.txt");
  // out_rhs.precision(17);
  // for (size_t i = 0; i < gridsize; ++i) {
  //   out_rhs << b[i] << " ";
  // }
  // out_rhs.close();

  std::cout << "Solving density SLE" << std::endl;
  solver->solve(*operation_mult, alpha, b, false, true);

  //TODO: remove this?
  // scale alphas to [0, 1] -> smaller function values, use?
  double max = alpha.max();
  double min = alpha.min();
  for (size_t i = 0; i < gridsize; i++) alpha[i] = alpha[i] * 1.0 / (max - min);

  // std::cout << "Creating regular grid to evaluate sparse grid density function on" << std::endl;
  // double h = 1.0 / std::pow(2.0, eval_grid_level);  // 2^-eval_grid_level
  // size_t dim_grid_points = 1 << eval_grid_level;
  // base::DataMatrix evaluationPoints(dim_grid_points * dim_grid_points, 2);
  // size_t linearIndex = 0;
  // for (size_t i = 0; i < dim_grid_points; i++) {
  //   for (size_t j = 0; j < dim_grid_points; j++) {
  //     double x = static_cast<double>(i) * h;
  //     double y = static_cast<double>(j) * h;
  //     evaluationPoints(linearIndex, 0) = x;
  //     evaluationPoints(linearIndex, 1) = y;
  //     linearIndex += 1;
  //   }
  // }

  // datadriven::OperationMultipleEvalConfiguration configuration(
  //     datadriven::OperationMultipleEvalType::STREAMING,
  //     datadriven::OperationMultipleEvalSubType::DEFAULT);

  // std::cout << "Creating multieval operation" << std::endl;
  // auto eval = std::unique_ptr<base::OperationMultipleEval>(
  //     op_factory::createOperationMultipleEval(*grid, evaluationPoints, configuration));

  // base::DataVector results(evaluationPoints.getNrows());
  // std::cout << "Evaluating at evaluation grid points" << std::endl;
  // eval->mult(alpha, results);

  std::cout << "Starting graph creation..." << std::endl;
  auto operation_graph =
      std::unique_ptr<sgpp::datadriven::DensityOCLMultiPlatform::OperationCreateGraphOCL>(
          sgpp::datadriven::createNearestNeighborGraphConfigured(trainingData, k, dimension,
                                                                 configFileName));
  std::vector<int> graph(trainingData.getNrows() * k);
  operation_graph->create_graph(graph);

  // std::ofstream out("graph_erg_dim2_depth11.txt");
  // for (size_t i = 0; i < trainingData.getNrows(); ++i) {
  //   for (size_t j = 0; j < k; ++j) {
  //     out << graph[i * k + j] << " ";
  //   }
  //   out << std::endl;
  // }
  // out.close();

  std::cout << "Pruning graph..." << std::endl;
  sgpp::datadriven::DensityOCLMultiPlatform::OperationPruneGraphOCL *operation_prune =
      sgpp::datadriven::pruneNearestNeighborGraphConfigured(*grid, dimension, alpha, trainingData,
                                                            threshold, k, configFileName);
  operation_prune->prune_graph(graph);

  // out.open("graph_pruned_erg_dim2_depth11.txt");
  // for (size_t i = 0; i < trainingData.getNrows(); ++i) {
  //   for (size_t j = 0; j < k; ++j) {
  //     out << graph[i * k + j] << " ";
  //   }
  //   out << std::endl;
  // }
  // out.close();
  std::cout << "Finding clusters..." << std::endl;
  std::vector<size_t> cluster_assignments =
      sgpp::datadriven::DensityOCLMultiPlatform::OperationCreateGraphOCL::find_clusters(graph, k);
  // out.open("cluster_erg.txt");
  // for (size_t datapoint : cluster_assignments) {
  //   out << datapoint << " ";
  // }
  // std::cout << results.toString();
  std::cout << std::endl << "all done!" << std::endl;
}
