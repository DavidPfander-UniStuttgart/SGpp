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
#include <sstream>
#include <string>
#include <vector>

#include "KNNFactory.hpp"
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
#include "sgpp/datadriven/operation/hash/OperationNearestNeighborSampled/OperationNearestNeighborSampled.hpp"
#include "sgpp/datadriven/operation/hash/OperationPruneGraphOCL/OpFactory.hpp"
#include "sgpp/datadriven/tools/ARFFTools.hpp"
#include "sgpp/globaldef.hpp"
#include "sgpp/solver/sle/ConjugateGradients.hpp"

double testAccuracy(const std::vector<int> correct,
                    const std::vector<int> result, const int size,
                    const int k) {
  std::cout << "size: " << size << std::endl;
  int count = 0;
  int total = 0;
  for (int s = 0; s < size; s += 1) {
    for (int c = 0; c < k; c += 1) {
      for (int r = 0; r < k; r += 1) {
        // if (correct[s * k + c] < 0 && result[s * k + r] < 0) {
        //   total += 1;
        //   continue;
        // }
        if (correct[s * k + c] == result[s * k + r]) {
          count += 1;
          break;
        }
      }
      total += 1;
    }
  }
  std::cout << "count: " << count << std::endl;
  std::cout << "total: " << total << std::endl;
  return static_cast<double>(count) / static_cast<double>(total);
}

double testDistanceAccuracy(const std::vector<double> data,
                            const std::vector<int> correct,
                            const std::vector<int> result, const int size,
                            const int dim, const int k) {
  double dist_sum_correct = 0.0;
  double dist_sum_lsh = 0.0;
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < k; ++j) {
      // if (correct[k * i + j] < 0) {
      //   continue;
      // }

      double dist_correct = 0.0;
      for (int d = 0; d < dim; ++d) {
        dist_correct +=
            (data[size * d + i] - data[size * d + correct[k * i + j]]) *
            (data[size * d + i] - data[size * d + correct[k * i + j]]);
      }
      dist_sum_correct += sqrt(dist_correct);
    }
    for (int j = 0; j < k; ++j) {
      // if (result[k * i + j] < 0) {
      //   continue;
      // }
      double dist_lsh = 0.0;
      for (int d = 0; d < dim; ++d) {
        dist_lsh += (data[size * d + i] - data[size * d + result[k * i + j]]) *
                    (data[size * d + i] - data[size * d + result[k * i + j]]);
      }
      dist_sum_lsh += sqrt(dist_lsh);
    }
  }
  return 1.0 - std::abs(dist_sum_lsh / dist_sum_correct);
}

std::vector<std::string> split(const std::string &s, char delim) {
  std::stringstream ss(s);
  std::string item;
  std::vector<std::string> r;

  while (std::getline(ss, item, delim)) {
    r.push_back(item);
  }

  return r;
}

using namespace sgpp;

int main(int argc, char **argv) {
  std::string datasetFileName;
  size_t eval_grid_level;
  size_t level;
  double lambda;
  std::string configFileName;
  uint64_t k;
  double threshold;

  bool write_density_grid;
  bool write_evaluated_density_full_grid;
  bool write_knn_graph;
  bool write_pruned_knn_graph;
  bool write_cluster_map;
  bool record_timings;
  std::string scenario_name;

  std::string compare_knn_csv_file_name;

  size_t refinement_steps;
  size_t refinement_points;
  size_t coarsening_points;
  double coarsening_threshold;
  std::string knn_algorithm;
  uint64_t lsh_tables;
  uint64_t lsh_hashes;
  double lsh_w;

  uint64_t sampling_chunk_size;
  uint64_t sampling_chunk_count;

  boost::program_options::options_description description("Allowed options");

  description.add_options()("help", "display help")(
      "datasetFileName",
      boost::program_options::value<std::string>(&datasetFileName),
      "training data set as an arff file")(
      "density_eval_full_grid_level",
      boost::program_options::value<size_t>(&eval_grid_level)->default_value(2),
      "level for the evaluation of the sparse grid density function on "
      "printable full grid")(
      "level", boost::program_options::value<size_t>(&level)->default_value(4),
      "level of the sparse grid used for density estimation")(
      "lambda",
      boost::program_options::value<double>(&lambda)->default_value(0.000001),
      "regularization for density estimation")(
      "config", boost::program_options::value<std::string>(&configFileName),
      "OpenCL and kernel configuration file")(
      "k", boost::program_options::value<uint64_t>(&k)->default_value(5),
      "specifies number of neighbors for kNN algorithm")(
      "threshold",
      boost::program_options::value<double>(&threshold)->default_value(0.0),
      "threshold for sparse grid function for removing edges")(
      "scenario_name",
      boost::program_options::value<std::string>(&scenario_name),
      "name for the current run, used when files are written")(
      "refinement_steps",
      boost::program_options::value<uint64_t>(&refinement_steps)
          ->default_value(0),
      "number of refinment steps for density estimation")(
      "refinement_points",
      boost::program_options::value<uint64_t>(&refinement_points)
          ->default_value(0),
      "number of points to refinement during density estimation")(
      "coarsen_points",
      boost::program_options::value<uint64_t>(&coarsening_points)
          ->default_value(0),
      "number of points to coarsen during density estimation")(
      "coarsen_threshold",
      boost::program_options::value<double>(&coarsening_threshold)
          ->default_value(1000.0),
      "for density estimation, only surpluses below threshold are "
      "coarsened")("knn_algorithm",
                   boost::program_options::value<std::string>(&knn_algorithm)
                       ->default_value("lsh"),
                   "type of kNN algorithm used, either 'lsh' or 'naive'")(
      "lsh_tables",
      boost::program_options::value<uint64_t>(&lsh_tables)->default_value(50),
      "number of hash tables for lsh knn")(
      "lsh_hashes",
      boost::program_options::value<uint64_t>(&lsh_hashes)->default_value(15),
      "number of hash functions used by lsh knn")(
      "lsh_w",
      boost::program_options::value<double>(&lsh_w)->default_value(1.5),
      "number of segments for hash functions used by lsh knn")(
      "write_knn_graph",
      boost::program_options::value<bool>(&write_knn_graph)
          ->default_value(false),
      "write the knn graph calculated to a csv-file")(
      "write_pruned_knn_graph",
      boost::program_options::value<bool>(&write_pruned_knn_graph)
          ->default_value(false),
      "write the pruned knn graph calculated to a csv-file")(
      "write_cluster_map",
      boost::program_options::value<bool>(&write_cluster_map)
          ->default_value(false),
      "write mapped clusters to a csv-file")(
      "write_density_grid",
      boost::program_options::value<bool>(&write_density_grid)
          ->default_value(false),
      "write the coordinates, levels and indices to a csv-file")(
      "write_evaluated_density_full_grid",
      boost::program_options::value<bool>(&write_evaluated_density_full_grid)
          ->default_value(false),
      "evaluate density function on full grid and write result to a "
      "csv-file")("record_timings",
                  boost::program_options::value<bool>(&record_timings)
                      ->default_value(false),
                  "write runtime performance measurements to a csv-file")(
      "compare_knn_csv_file_name",
      boost::program_options::value<std::string>(&compare_knn_csv_file_name),
      "compare the knn results to a reference solution")(
      "sampling_chunk_size",
      boost::program_options::value<uint64_t>(&sampling_chunk_size)
          ->default_value(0),
      "size of the chunk for sampling of dataset for knn")(
      "sampling_chunk_count",
      boost::program_options::value<uint64_t>(&sampling_chunk_count)
          ->default_value(1),
      "number of chunks for sampling of dataset for knn");

  boost::program_options::variables_map variables_map;

  boost::program_options::parsed_options options =
      parse_command_line(argc, argv, description);
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
      std::cerr << "error: dataset file does not exist: " << datasetFileName
                << std::endl;
      return 1;
    }
    std::cout << "datasetFileName: " << datasetFileName << std::endl;
  }

  // if (variables_map.count("eval_grid_level") == 0) {
  //   std::cerr << "error: option \"eval_grid_level\" not specified" <<
  //   std::endl; return 1;
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
      std::cerr << "error: config file does not exist: " << configFileName
                << std::endl;
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

  if (knn_algorithm.compare("lsh") == 0) {
    std::cout << "using lsh knn" << std::endl;
  } else if (knn_algorithm.compare("naive") == 0) {
    std::cout << "using naive multicore knn" << std::endl;
  } else if (knn_algorithm.compare("ocl") == 0) {
    std::cout << "using naive ocl knn" << std::endl;
  } else {
    std::cerr << "error: option \"knn_algorithm\" only supports 'lsh', 'ocl' "
                 "and 'naive'"
              << std::endl;
  }

  std::ofstream result_timings;
  if (record_timings) {
    std::cout << "output scenario name: " << scenario_name << std::endl;
    std::string result_timings_file_name(scenario_name + "_result_timings.csv");
    std::experimental::filesystem::path result_timings_path(
        result_timings_file_name);
    if (std::experimental::filesystem::exists(result_timings_path)) {
      result_timings.open(std::string("results/") + scenario_name +
                              "_result_timings.csv",
                          std::ios::out | std::ios::app);
    } else {
      result_timings.open(std::string("results/") + scenario_name +
                              "_result_timings.csv",
                          std::ios::out);
      result_timings
          << "dataset; grid_level; lambda; threshold; k; config; "
             "refine_steps; "
             "refine_points; coarsen_points; coarsen_threshold; "
             "duration_generate_b; "
             "gflops_generate_b; duration_density_average; "
             "gflops_density_average; "
             "duration_create_graph; gflops_create_graph; "
             "duration_prune_graph; "
             "gflops_prune_graph; total_duration_without_disk;total_duration"
          << std::endl;
    }
    result_timings << datasetFileName << "; " << level << "; " << lambda << "; "
                   << threshold << "; " << k << "; " << configFileName << ";"
                   << refinement_steps << "; " << refinement_points << "; "
                   << coarsening_points << "; " << coarsening_threshold << "; ";
  }

  //   size_t refinement_steps;
  // size_t refinement_points;
  // size_t coarsening_points;
  // double coarsening_threshold;

  // configure refinement
  // sgpp::base::AdpativityConfiguration adaptConfig;
  // adaptConfig.maxLevelType_ = false;
  // adaptConfig.noPoints_ = 80;
  // adaptConfig.numRefinements_ = 0;
  // adaptConfig.percent_ = 200.0;
  // adaptConfig.threshold_ = 0.0;

  std::chrono::time_point<std::chrono::system_clock> total_timer_start =
      std::chrono::system_clock::now();

  // read dataset
  std::cout << "reading dataset...";
  datadriven::Dataset dataset =
      datadriven::ARFFTools::readARFF(datasetFileName);
  std::cout << "done" << std::endl;

  std::chrono::time_point<std::chrono::system_clock>
      total_timer_start_without_disk = std::chrono::system_clock::now();

  size_t dimension = dataset.getDimension();
  std::cout << "dimension: " << dimension << std::endl;

  base::DataMatrix &trainingData = dataset.getData();
  std::cout << "data points: " << trainingData.getNrows() << std::endl;

  // create grid
  std::unique_ptr<base::Grid> grid(base::Grid::createLinearGrid(dimension));
  base::GridGenerator &grid_generator = grid->getGenerator();
  grid_generator.regular(level);
  std::cout << "Initial grid created! Number of grid points: "
            << grid->getSize() << std::endl;

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
    std::unique_ptr<datadriven::DensityOCLMultiPlatform::OperationDensity>
        operation_mult(datadriven::createDensityOCLMultiPlatformConfigured(
            *grid, dimension, lambda, configFileName));

    std::cout << "Calculating right-hand side" << std::endl;

    operation_mult->generateb(trainingData, b);

    double last_duration_generate_b = operation_mult->getLastDurationB();
    std::cout << "last_duration_generate_b: " << last_duration_generate_b << "s"
              << std::endl;
    double ops_generate_b =
        static_cast<double>(grid->getSize() * trainingData.getNrows() *
                            (6 * dimension + 1)) *
        1E-9;
    std::cout << "ops_generate_b: " << ops_generate_b << std::endl;
    double flops_generate_b = ops_generate_b / last_duration_generate_b;
    std::cout << "flops_generate_b: " << flops_generate_b << " GFLOPS"
              << std::endl;

    result_timings << last_duration_generate_b << "; " << flops_generate_b
                   << "; ";

    std::cout << "Solving density SLE" << std::endl;
    solver->solve(*operation_mult, alpha, b, false, true);

    double acc_duration_density = operation_mult->getAccDurationDensityMult();
    std::cout << "acc_duration_density: " << acc_duration_density << "s"
              << std::endl;

    size_t iterations = solver->getNumberIterations();
    size_t act_it = iterations + 1 + (iterations / 50);
    std::cout << "act_it: " << act_it << std::endl;
    double ops_density = static_cast<double>(std::pow(grid->getSize(), 2) *
                                             act_it * (14 * dimension + 2)) *
                         1E-9;
    std::cout << "ops_density: " << ops_density << " GOps" << std::endl;
    double flops_density = ops_density / acc_duration_density;
    std::cout << "flops_density: " << flops_density << " GFLOPS" << std::endl;

    result_timings << acc_duration_density << "; " << flops_density << "; ";
  }

  for (size_t i = 0; i < refinement_steps; i++) {
    if (refinement_points > 0) {
      sgpp::base::SurplusRefinementFunctor refine_func(alpha,
                                                       refinement_points);
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

      std::unique_ptr<datadriven::DensityOCLMultiPlatform::OperationDensity>
          operation_mult(datadriven::createDensityOCLMultiPlatformConfigured(
              *grid, dimension, lambda, configFileName));

      operation_mult->generateb(trainingData, b);

      double last_duration_generate_b = operation_mult->getLastDurationB();
      std::cout << "last_duration_generate_b: " << last_duration_generate_b
                << "s" << std::endl;
      double ops_generate_b =
          static_cast<double>(grid->getSize() * trainingData.getNrows() *
                              (6 * dimension + 1)) *
          1E-9;
      std::cout << "ops_generate_b: " << ops_generate_b << std::endl;
      double flops_generate_b = ops_generate_b / last_duration_generate_b;
      std::cout << "flops_generate_b: " << flops_generate_b << " GFLOPS"
                << std::endl;

      std::cout << "Solving density SLE" << std::endl;
      solver->solve(*operation_mult, alpha, b, false, true);

      std::cout << "Grid points after refinement step: " << grid->getSize()
                << std::endl;

      size_t iterations = solver->getNumberIterations();
      size_t act_it = iterations + 1 + (iterations / 50);
      std::cout << "act_it: " << act_it << std::endl;

      double acc_duration_density = operation_mult->getAccDurationDensityMult();
      std::cout << "acc_duration_density: " << acc_duration_density << "s"
                << std::endl;
      double ops_density = static_cast<double>(std::pow(grid->getSize(), 2) *
                                               act_it * (14 * dimension + 2)) *
                           1E-9;
      std::cout << "ops_density: " << ops_density << " GOps" << std::endl;
      double flops_density = ops_density / acc_duration_density;
      std::cout << "flops_density: " << flops_density << " GFLOPS" << std::endl;
    }

    if (coarsening_points > 0) {
      size_t grid_size_before_coarsen = grid->getSize();
      sgpp::base::SurplusCoarseningFunctor coarsen_func(
          alpha, coarsening_points, coarsening_threshold);
      grid_generator.coarsen(coarsen_func, alpha);

      size_t grid_size_after_coarsen = grid->getSize();
      std::cout << "coarsen: removed "
                << (grid_size_before_coarsen - grid_size_after_coarsen)
                << " grid points" << std::endl;

      // adjust alpha to coarsen grid
      alpha.resize(grid->getSize());

      std::unique_ptr<datadriven::DensityOCLMultiPlatform::OperationDensity>
          operation_mult(datadriven::createDensityOCLMultiPlatformConfigured(
              *grid, dimension, lambda, configFileName));

      // regenerate b with coarsen grid
      b.resize(grid->getSize());

      // TODO: remove this?
      operation_mult->generateb(trainingData, b);

      double last_duration_generate_b = operation_mult->getLastDurationB();
      std::cout << "last_duration_generate_b: " << last_duration_generate_b
                << "s" << std::endl;
      double ops_generate_b =
          static_cast<double>(grid->getSize() * trainingData.getNrows() *
                              (6 * dimension + 1)) *
          1E-9;
      std::cout << "ops_generate_b: " << ops_generate_b << std::endl;
      double flops_generate_b = ops_generate_b / last_duration_generate_b;
      std::cout << "flops_generate_b: " << flops_generate_b << " GFLOPS"
                << std::endl;

      std::cout << "Solving density SLE" << std::endl;
      solver->solve(*operation_mult, alpha, b, false, true);

      std::cout << "Grid points after coarsening step: " << grid->getSize()
                << std::endl;

      size_t iterations = solver->getNumberIterations();
      size_t act_it = iterations + 1 + (iterations / 50);
      std::cout << "act_it: " << act_it << std::endl;

      double acc_duration_density = operation_mult->getAccDurationDensityMult();
      std::cout << "acc_duration_density: " << acc_duration_density << "s"
                << std::endl;

      double ops_density = static_cast<double>(std::pow(grid->getSize(), 2) *
                                               act_it * (14 * dimension + 2)) *
                           1E-9;

      std::cout << "ops_density: " << ops_density << " GOps" << std::endl;
      double flops_density = ops_density / acc_duration_density;
      std::cout << "flops_density: " << flops_density << " GFLOPS" << std::endl;
    }
  }

  density_timer_stop = std::chrono::system_clock::now();
  std::chrono::duration<double> density_elapsed_seconds =
      density_timer_stop - density_timer_start;
  std::cout << "density_duration_total: " << density_elapsed_seconds.count()
            << std::endl;

  // scale alphas to [0, 1] -> smaller function values, use?
  double max = alpha.max();
  double min = alpha.min();
  for (size_t i = 0; i < grid->getSize(); i++) {
    alpha[i] = alpha[i] * 1.0 / (max - min);
  }

  if (write_density_grid) {
    std::ofstream out_grid(std::string("results/") + scenario_name +
                           "_grid_coord.csv");
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

    out_grid = std::ofstream(std::string("results/") + scenario_name +
                             "_grid_levels.csv");
    for (size_t j = 0; j < grid->getSize(); j++) {
      sgpp::base::GridPoint point = storage.getPoint(j);
      sgpp::base::GridPoint::level_type l;
      sgpp::base::GridPoint::index_type i;
      for (size_t d = 0; d < dimension; d++) {
        point.get(d, l, i);
        if (d > 0) {
          out_grid << ", ";
        }
        out_grid << l;
      }
      out_grid << std::endl;
    }
    out_grid.close();
    out_grid = std::ofstream(std::string("results/") + scenario_name +
                             "_grid_indices.csv");
    for (size_t j = 0; j < grid->getSize(); j++) {
      sgpp::base::GridPoint point = storage.getPoint(j);
      sgpp::base::GridPoint::level_type l;
      sgpp::base::GridPoint::index_type i;
      for (size_t d = 0; d < dimension; d++) {
        point.get(d, l, i);
        if (d > 0) {
          out_grid << ", ";
        }
        out_grid << i;
      }
      out_grid << std::endl;
    }
    out_grid.close();
  }

  if (write_evaluated_density_full_grid) {
    std::cout << "Creating regular grid to evaluate sparse grid density "
                 "function on..."
              << std::endl;
    double h = 1.0 / std::pow(2.0, eval_grid_level); // 2^-eval_grid_level
    size_t dim_grid_points = (1 << eval_grid_level) + 1;
    size_t total_grid_points = 1;
    for (size_t d = 0; d < dimension; d++) {
      total_grid_points *= dim_grid_points;
    }
    std::cout << "total density evaluation full grid grid points: "
              << total_grid_points << std::endl;
    if (total_grid_points > 1E6) {
      std::cerr << "warning: density full grid evaluation might take very "
                   "long, this is a potential input error"
                << std::endl;
    }

    base::DataMatrix evaluationPoints(total_grid_points, dimension);

    size_t linearIndex = 0;
    std::vector<double> eval_point_enum(dimension);
    for (size_t d = 0; d < dimension; d++) {
      eval_point_enum[d] = 0;
    }
    // do first point seperately
    for (size_t d = 0; d < dimension; d++) {
      double x = static_cast<double>(eval_point_enum[d]) * h;
      evaluationPoints(linearIndex, d) = x;
    }
    linearIndex += 1;

    size_t dim_index = 0;
    while (dim_index < dimension) {
      if (eval_point_enum[dim_index] + 1 < dim_grid_points) {
        eval_point_enum[dim_index] += 1;
        for (size_t d = 0; d < dim_index; d++) {
          eval_point_enum[d] = 0;
        }
        dim_index = 0;

        for (size_t d = 0; d < dimension; d++) {
          double x = static_cast<double>(eval_point_enum[d]) * h;
          evaluationPoints(linearIndex, d) = x;
        }
        linearIndex += 1;
      } else {
        dim_index += 1;
      }
    }

    datadriven::OperationMultipleEvalConfiguration configuration(
        datadriven::OperationMultipleEvalType::STREAMING,
        datadriven::OperationMultipleEvalSubType::DEFAULT);

    std::cout << "Creating multieval operation" << std::endl;
    std::unique_ptr<base::OperationMultipleEval> eval(
        op_factory::createOperationMultipleEval(*grid, evaluationPoints,
                                                configuration));

    base::DataVector results(evaluationPoints.getNrows());
    std::cout << "Evaluating at evaluation grid points" << std::endl;
    eval->mult(alpha, results);

    std::ofstream out_density(std::string("results/") + scenario_name +
                              std::string("_density_eval.csv"));
    out_density.precision(20);
    for (size_t eval_index = 0; eval_index < evaluationPoints.getNrows();
         eval_index += 1) {
      for (size_t d = 0; d < dimension; d += 1) {
        out_density << evaluationPoints[eval_index * dimension + d] << ", ";
      }
      out_density << results[eval_index];
      out_density << std::endl;
    }
    out_density.close();
  }

  std::vector<int64_t> graph;

  if (sampling_chunk_size == 0) {
    sampling_chunk_size = trainingData.getNrows();
  }

  sgpp::datadriven::OperationNearestNeighborSampled knn_op(true);

  {
    std::cout << "Starting graph creation..." << std::endl;

    if (knn_algorithm.compare("lsh") == 0) {
      graph = knn_op.knn_lsh(dimension, trainingData, k, lsh_tables, lsh_hashes,
                             lsh_w);
      double lsh_duration = knn_op.get_last_duration();

      std::cout << "lsh_duration: " << lsh_duration << "s" << std::endl;

      result_timings << lsh_duration << "; 0.0;";
    } else if (knn_algorithm.compare("ocl") == 0) {
      graph = knn_op.knn_ocl(dimension, trainingData, k, configFileName);
      double last_duration_create_graph = knn_op.get_last_duration();
      std::cout << "last_duration_create_graph: " << last_duration_create_graph
                << "s" << std::endl;

      double ops_create_graph =
          static_cast<double>(std::pow(trainingData.getNrows(), 2) * 4 *
                              dimension) *
          1E-9;
      std::cout << "ops_create_graph: " << ops_create_graph << " GOps"
                << std::endl;
      double flops_create_graph = ops_create_graph / last_duration_create_graph;
      std::cout << "flops_create_graph: " << flops_create_graph << " GFLOPS"
                << std::endl;

      result_timings << last_duration_create_graph << "; " << flops_create_graph
                     << "; ";
    } else if (knn_algorithm.compare("naive") == 0) {

      graph = knn_op.knn_naive(dimension, trainingData, k);
      double naive_duration = knn_op.get_last_duration();
      std::cout << "naive_duration: " << naive_duration << "s" << std::endl;
      result_timings << naive_duration << "; 0.0;";
    }

    if (write_knn_graph) {
      knn_op.write_graph_file(std::string("results/") + scenario_name +
                                  "_graph_naive.csv",
                              graph, k);
    }

    if (variables_map.count("compare_knn_csv_file_name") > 0) {
      std::vector<int64_t> neighbors_reference =
          knn_op.read_csv(compare_knn_csv_file_name);
      std::vector<int64_t> graph_converted;
      graph_converted = std::vector<int64_t>(graph.begin(), graph.end());
      double acc_assigned = knn_op.test_accuracy(
          neighbors_reference, graph_converted, trainingData.getNrows(), k);
      double acc_distance = knn_op.test_distance_accuracy(
          trainingData, neighbors_reference, graph_converted,
          trainingData.getNrows(), dimension, k);
      std::cout << "knn correctly assigned: " << acc_assigned << std::endl;
      std::cout << "knn distance error: " << acc_distance << std::endl;
    }
  }

  {
    std::cout << "Pruning graph..." << std::endl;

    std::unique_ptr<
        sgpp::datadriven::DensityOCLMultiPlatform::OperationPruneGraphOCL>
        operation_prune(sgpp::datadriven::pruneNearestNeighborGraphConfigured(
            *grid, dimension, alpha, trainingData, threshold, k,
            configFileName));
    std::vector<int> graph_unconverted;
    operation_prune->prune_graph(graph_unconverted);
    // TODO: remove after operation is converted to int64_t
    graph = std::vector<int64_t>(graph_unconverted.begin(),
                                 graph_unconverted.end());

    double last_duration_prune_graph = operation_prune->getLastDuration();
    std::cout << "last_duration_prune_graph: " << last_duration_prune_graph
              << "s" << std::endl;

    // middlepoint between node and neighbor ops
    double ops_prune_graph =
        static_cast<double>(trainingData.getNrows() * grid->getSize() *
                            (k + 1) * (6 * dimension + 2)) *
        1E-9;
    std::cout << "ops_prune_graph: " << ops_prune_graph << " GOps" << std::endl;
    double flops_prune_graph = ops_prune_graph / last_duration_prune_graph;
    std::cout << "flops_prune_graph: " << flops_prune_graph << " GFLOPS"
              << std::endl;

    result_timings << last_duration_prune_graph << "; " << flops_prune_graph
                   << "; ";

    if (write_pruned_knn_graph) {
      std::vector<int64_t> graph_converted;
      graph_converted = std::vector<int64_t>(graph.begin(), graph.end());
      knn_op.write_graph_file(std::string("results/") + scenario_name +
                                  "_graph_pruned.csv",
                              graph_converted, k);
    }
  }

  {
    std::cout << "Finding clusters..." << std::endl;
    std::vector<int> node_cluster_map;
    sgpp::datadriven::DensityOCLMultiPlatform::OperationCreateGraphOCL::
        neighborhood_list_t clusters;

    std::chrono::time_point<std::chrono::system_clock> find_cluster_timer_start;
    std::chrono::time_point<std::chrono::system_clock> find_cluster_timer_stop;
    find_cluster_timer_start = std::chrono::system_clock::now();

    // TODO: update after conversion
    std::vector<int32_t> graph_converted(graph.begin(), graph.end());

    sgpp::datadriven::DensityOCLMultiPlatform::OperationCreateGraphOCL::
        find_clusters(graph_converted, k, node_cluster_map, clusters);

    find_cluster_timer_stop = std::chrono::system_clock::now();
    std::chrono::duration<double> find_cluster_elapsed_seconds =
        find_cluster_timer_stop - find_cluster_timer_start;
    std::cout << "find_cluster_duration_total: "
              << find_cluster_elapsed_seconds.count() << std::endl;

    std::cout << "detected clusters: " << clusters.size() << std::endl;

    if (write_cluster_map) {
      std::ofstream out_cluster_map(std::string("results/") + scenario_name +
                                    "_cluster_map.csv");
      for (size_t i = 0; i < trainingData.getNrows(); ++i) {
        out_cluster_map << node_cluster_map[i] << std::endl;
      }
      out_cluster_map.close();
    }
  }

  {
    auto total_timer_stop = std::chrono::system_clock::now();
    std::chrono::duration<double> total_elapsed_seconds =
        total_timer_stop - total_timer_start;
    std::chrono::duration<double> total_duration_without_disk_elapsed_seconds =
        total_timer_stop - total_timer_start_without_disk;
    double total_duration = total_elapsed_seconds.count();
    double total_duration_without_disk =
        total_duration_without_disk_elapsed_seconds.count();
    std::cout << "total_duration_without_disk: " << total_duration_without_disk
              << std::endl;
    std::cout << "total_duration: " << total_duration << std::endl;
    result_timings << total_duration_without_disk << "; " << total_duration
                   << std::endl;
  }
  std::cout << std::endl << "all done!" << std::endl;
}
