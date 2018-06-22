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
  std::string datasetFileName;
  size_t eval_grid_level;
  size_t level;
  double lambda;
  std::string configFileName;
  uint64_t k;
  double threshold;

  bool do_output_graphs = false;
  std::string scenario_name;

  size_t refinement_steps;
  size_t refinement_points;
  size_t coarsening_points;
  double coarsening_threshold;

  boost::program_options::options_description description("Allowed options");

  description.add_options()("help", "display help")(
      "datasetFileName", boost::program_options::value<std::string>(&datasetFileName),
      "training data set as an arff file")(
      "density_eval_full_grid_level",
      boost::program_options::value<size_t>(&eval_grid_level)->default_value(2),
      "level for the evaluation of the sparse grid density function on printable full grid")(
      "level", boost::program_options::value<size_t>(&level)->default_value(4),
      "level of the sparse grid used for density estimation")(
      "lambda", boost::program_options::value<double>(&lambda)->default_value(0.000001),
      "regularization for density estimation")(
      "config", boost::program_options::value<std::string>(&configFileName),
      "OpenCL and kernel configuration file")(
      "k", boost::program_options::value<uint64_t>(&k)->default_value(5),
      "specifies number of neighbors for kNN algorithm")(
      "threshold", boost::program_options::value<double>(&threshold)->default_value(0.0),
      "threshold for sparse grid function for removing edges")(
      "write_graphs", boost::program_options::value<std::string>(&scenario_name),
      "output the clustering steps into files")(
      "refinement_steps",
      boost::program_options::value<uint64_t>(&refinement_steps)->default_value(0),
      "number of refinment steps for density estimation")(
      "refinement_points",
      boost::program_options::value<uint64_t>(&refinement_points)->default_value(0),
      "number of points to refinement during density estimation")(
      "coarsen_points",
      boost::program_options::value<uint64_t>(&coarsening_points)->default_value(0),
      "number of points to coarsen during density estimation")(
      "coarsen_threshold",
      boost::program_options::value<double>(&coarsening_threshold)->default_value(1000.0),
      "for density estimation, only surpluses below threshold are coarsened");

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

  // if (variables_map.count("eval_grid_level") == 0) {
  //   std::cerr << "error: option \"eval_grid_level\" not specified" << std::endl;
  //   return 1;
  // } else {
  //   std::cout << "eval_grid_level: " << eval_grid_level << std::endl;
  // }

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

  if (variables_map.count("write_graphs") == 1) {
    do_output_graphs = true;
    std::cout << "output scenario name: " << scenario_name << std::endl;
  }

  // configure refinement
  // sgpp::base::AdpativityConfiguration adaptConfig;
  // adaptConfig.maxLevelType_ = false;
  // adaptConfig.noPoints_ = 80;
  // adaptConfig.numRefinements_ = 0;
  // adaptConfig.percent_ = 200.0;
  // adaptConfig.threshold_ = 0.0;

  // read dataset
  std::cout << "reading dataset...";
  datadriven::Dataset dataset = datadriven::ARFFTools::readARFF(datasetFileName);
  std::cout << "done" << std::endl;
  size_t dimension = dataset.getDimension();
  std::cout << "dimension: " << dimension << std::endl;

  if (dimension != 2 && do_output_graphs) {
    std::cerr << "error: write_graphs-option is only available for 2d problems" << std::endl;
    return 1;
  }

  base::DataMatrix &trainingData = dataset.getData();
  std::cout << "data points: " << trainingData.getNrows() << std::endl;

  // create grid
  std::unique_ptr<base::Grid> grid(base::Grid::createLinearGrid(dimension));
  base::GridGenerator &grid_generator = grid->getGenerator();
  grid_generator.regular(level);
  std::cout << "Initial grid created! Number of grid points: " << grid->getSize() << std::endl;

  // create solver
  std::chrono::time_point<std::chrono::system_clock> density_timer_start;
  std::chrono::time_point<std::chrono::system_clock> density_timer_stop;
  density_timer_start = std::chrono::system_clock::now();
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

    std::cout << "Solving density SLE" << std::endl;
    solver->solve(*operation_mult, alpha, b, false, true);

    std::cout << "acc_duration_b: " << operation_mult->getAccDurationB() << std::endl;
    double acc_duration_density = operation_mult->getAccDurationDensityMult();
    std::cout << "acc_duration_density: " << acc_duration_density << std::endl;

    double ops_density = std::pow(static_cast<double>(grid->getSize()), 2.0) *
                         (12.0 * static_cast<double>(dimension) + 2.0) * 1E-9;
    std::cout << "ops_density: " << ops_density << " GOps" << std::endl;
    double flops_density = ops_density / acc_duration_density;
    std::cout << "flops_density: " << flops_density << " GFLOPS" << std::endl;
  }

  for (size_t i = 0; i < refinement_steps; i++) {
    if (refinement_points > 0) {
      sgpp::base::SurplusRefinementFunctor refine_func(alpha, refinement_points);
      grid_generator.refine(refine_func);
      size_t old_size = alpha.getSize();

      // adjust alpha to refined grid
      alpha.resize(grid->getSize());
      for (size_t j = old_size; j < alpha.getSize(); j++) {
        alpha[j] = 0.0;
      }

      // regenerate b with refined grid
      b.resize(grid->getSize());
      for (size_t j = old_size; j < alpha.getSize(); j++) {
        b[j] = 0.0;
      }

      std::unique_ptr<datadriven::DensityOCLMultiPlatform::OperationDensity> operation_mult(
          datadriven::createDensityOCLMultiPlatformConfigured(*grid, dimension, lambda,
                                                              configFileName));

      operation_mult->generateb(trainingData, b);

      std::cout << "Solving density SLE" << std::endl;
      solver->solve(*operation_mult, alpha, b, false, true);

      std::cout << "Grid points after refinement step: " << grid->getSize() << std::endl;

      std::cout << "acc_duration_b: " << operation_mult->getAccDurationB() << std::endl;
      double acc_duration_density = operation_mult->getAccDurationDensityMult();
      std::cout << "acc_duration_density: " << acc_duration_density << std::endl;
      double ops_density = std::pow(static_cast<double>(grid->getSize()), 2.0) *
                           (12.0 * static_cast<double>(dimension) + 2.0) * 1E-9;
      std::cout << "ops_density: " << ops_density << " GOps" << std::endl;
      double flops_density = ops_density / acc_duration_density;
      std::cout << "flops_density: " << flops_density << " GFLOPS" << std::endl;
    }

    if (coarsening_points > 0) {
      size_t grid_size_before_coarsen = grid->getSize();
      sgpp::base::SurplusCoarseningFunctor coarsen_func(alpha, coarsening_points,
                                                        coarsening_threshold);
      grid_generator.coarsen(coarsen_func, alpha);

      size_t grid_size_after_coarsen = grid->getSize();
      std::cout << "coarsen: removed " << (grid_size_before_coarsen - grid_size_after_coarsen)
                << " grid points" << std::endl;

      // adjust alpha to coarsen grid
      alpha.resize(grid->getSize());

      std::unique_ptr<datadriven::DensityOCLMultiPlatform::OperationDensity> operation_mult(
          datadriven::createDensityOCLMultiPlatformConfigured(*grid, dimension, lambda,
                                                              configFileName));

      // regenerate b with coarsen grid
      b.resize(grid->getSize());

      // TODO: remove this?
      operation_mult->generateb(trainingData, b);

      std::cout << "Solving density SLE" << std::endl;
      solver->solve(*operation_mult, alpha, b, false, true);

      std::cout << "Grid points after coarsening step: " << grid->getSize() << std::endl;

      std::cout << "acc_duration_b: " << operation_mult->getAccDurationB() << std::endl;
      double acc_duration_density = operation_mult->getAccDurationDensityMult();
      std::cout << "acc_duration_density: " << acc_duration_density << std::endl;
      double ops_density = std::pow(static_cast<double>(grid->getSize()), 2.0) *
                           (12.0 * static_cast<double>(dimension) + 2.0) * 1E-9;
      std::cout << "ops_density: " << ops_density << " GOps" << std::endl;
      double flops_density = ops_density / acc_duration_density;
      std::cout << "flops_density: " << flops_density << " GFLOPS" << std::endl;
    }
  }

  density_timer_stop = std::chrono::system_clock::now();
  std::chrono::duration<double> density_elapsed_seconds = density_timer_stop - density_timer_start;
  std::cout << "density_duration_total: " << density_elapsed_seconds.count() << std::endl;

  // scale alphas to [0, 1] -> smaller function values, use?
  double max = alpha.max();
  double min = alpha.min();
  for (size_t i = 0; i < grid->getSize(); i++) {
    alpha[i] = alpha[i] * 1.0 / (max - min);
  }

  if (do_output_graphs) {
    std::ofstream out_grid(scenario_name + "_grid.csv");
    auto &storage = grid->getStorage();
    for (size_t i = 0; i < grid->getSize(); i++) {
      sgpp::base::HashGridPoint point = storage.getPoint(i);
      for (size_t d = 0; d < dimension; d++) {
        if (d > 0) {
          out_grid << ", ";
        }
        out_grid << storage.getCoordinate(point, d);
      }
      out_grid << std::endl;
    }
    out_grid.close();
  }

  if (do_output_graphs) {
    std::cout << "Creating regular grid to evaluate sparse grid density function on..."
              << std::endl;
    double h = 1.0 / std::pow(2.0, eval_grid_level);  // 2^-eval_grid_level
    size_t dim_grid_points = (1 << eval_grid_level) + 1;
    base::DataMatrix evaluationPoints(dim_grid_points * dim_grid_points, 2);
    size_t linearIndex = 0;
    for (size_t i = 0; i < dim_grid_points; i++) {
      for (size_t j = 0; j < dim_grid_points; j++) {
        double x = static_cast<double>(i) * h;
        double y = static_cast<double>(j) * h;
        evaluationPoints(linearIndex, 0) = x;
        evaluationPoints(linearIndex, 1) = y;
        linearIndex += 1;
      }
    }

    datadriven::OperationMultipleEvalConfiguration configuration(
        datadriven::OperationMultipleEvalType::STREAMING,
        datadriven::OperationMultipleEvalSubType::DEFAULT);

    std::cout << "Creating multieval operation" << std::endl;
    std::unique_ptr<base::OperationMultipleEval> eval(
        op_factory::createOperationMultipleEval(*grid, evaluationPoints, configuration));

    base::DataVector results(evaluationPoints.getNrows());
    std::cout << "Evaluating at evaluation grid points" << std::endl;
    eval->mult(alpha, results);

    std::ofstream out_density(scenario_name + std::string("_density_eval.csv"));
    out_density.precision(20);
    for (size_t eval_index = 0; eval_index < evaluationPoints.getNrows(); eval_index += 1) {
      for (size_t d = 0; d < dimension; d += 1) {
        out_density << evaluationPoints[eval_index * dimension + d] << ", ";
      }
      out_density << results[eval_index];
      out_density << std::endl;
    }
    out_density.close();
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

    // for (size_t i = 0; i < graph.size() / k; i++)  {
    //   std::cout << " node: " << i << " neigh: ";
    //   for (size_t cur_k = 0; cur_k < k; cur_k+= 1) {
    //    if (cur_k > 0) {
    //      std::cout << ", ";
    //    }
    //    std::cout << graph[i * k + cur_k];
    //   }
    //   std::cout << std::endl;
    // }

    if (do_output_graphs) {
      std::ofstream out_graph(scenario_name + "_graph.csv");
      for (size_t i = 0; i < trainingData.getNrows(); ++i) {
        bool first = true;
        for (size_t j = 0; j < k; ++j) {
          if (graph[i * k + j] == -1) {
            continue;
          }
          if (!first) {
            out_graph << ", ";
          } else {
            first = false;
          }
          out_graph << graph[i * k + j];
        }
        out_graph << std::endl;
      }
      out_graph.close();
    }
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
                             static_cast<double>(grid->getSize()) * static_cast<double>(k) *
                             (8.0 * static_cast<double>(dimension) + 2) * 1E-9;
    ops_prune_graph += static_cast<double>(trainingData.getNrows()) *
                       static_cast<double>(grid->getSize()) *
                       (static_cast<double>(dimension) * 5 + 2) * 1E-9;
    std::cout << "ops_prune_graph: " << ops_prune_graph << " GOps" << std::endl;
    double flops_prune_graph = ops_prune_graph / last_duration_prune_graph;
    std::cout << "flops_prune_graph: " << flops_prune_graph << " GFLOPS" << std::endl;

    if (do_output_graphs) {
      std::ofstream out_graph(scenario_name + "_graph_pruned.csv");
      for (size_t i = 0; i < trainingData.getNrows(); ++i) {
        bool first = true;
        for (size_t j = 0; j < k; ++j) {
          if (graph[i * k + j] == -1 || graph[i * k + j] == -2) {
            continue;
          }
          if (!first) {
            out_graph << ", ";
          } else {
            first = false;
          }
          out_graph << graph[i * k + j];
        }
        out_graph << std::endl;
      }
      out_graph.close();
    }
  }

  {
    std::cout << "Finding clusters..." << std::endl;
    std::vector<int> node_cluster_map;
    sgpp::datadriven::DensityOCLMultiPlatform::OperationCreateGraphOCL::neighborhood_list_t
        clusters;

    std::chrono::time_point<std::chrono::system_clock> find_cluster_timer_start;
    std::chrono::time_point<std::chrono::system_clock> find_cluster_timer_stop;
    find_cluster_timer_start = std::chrono::system_clock::now();

    sgpp::datadriven::DensityOCLMultiPlatform::OperationCreateGraphOCL::find_clusters(
        graph, k, node_cluster_map, clusters);

    find_cluster_timer_stop = std::chrono::system_clock::now();
    std::chrono::duration<double> find_cluster_elapsed_seconds =
        find_cluster_timer_stop - find_cluster_timer_start;
    std::cout << "find_cluster_duration_total: " << find_cluster_elapsed_seconds.count()
              << std::endl;

    std::cout << "detected clusters: " << clusters.size() << std::endl;

    if (do_output_graphs) {
      std::ofstream out_cluster_map(scenario_name + "_cluster_map.csv");
      for (size_t i = 0; i < trainingData.getNrows(); ++i) {
        out_cluster_map << node_cluster_map[i] << std::endl;
      }
      out_cluster_map.close();
    }
  }

  std::cout << std::endl << "all done!" << std::endl;
}
