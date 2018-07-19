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

#include <sgpp/base/grid/storage/hashmap/HashGridPoint.hpp>

using namespace sgpp;

int main(int argc, char **argv) {
  std::string datasetFileName;
  size_t eval_grid_level;
  size_t base_dim;
  size_t base_dim_level;
  double lambda;
  std::string configFileName;
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
      "base_dim", boost::program_options::value<size_t>(&base_dim)->default_value(4),
      "base dim of the reduced-dimension sparse grid")(
      "base_dim_level", boost::program_options::value<size_t>(&base_dim_level)->default_value(4),
      "base level of the reduced-dimension sparse grid")(
      "lambda", boost::program_options::value<double>(&lambda)->default_value(0.000001),
      "regularization for density estimation")(
      "config", boost::program_options::value<std::string>(&configFileName),
      "OpenCL and kernel configuration file")(
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

  if (variables_map.count("base_dim") == 0) {
    std::cerr << "error: option \"base_dim\" not specified" << std::endl;
    return 1;
  } else {
    std::cout << "base_dim: " << base_dim << std::endl;
  }

  if (variables_map.count("base_dim_level") == 0) {
    std::cerr << "error: option \"base_dim_level\" not specified" << std::endl;
    return 1;
  } else {
    std::cout << "base_dim_level: " << base_dim_level << std::endl;
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

  std::chrono::time_point<std::chrono::system_clock> total_timer_start =
      std::chrono::system_clock::now();

  // create d4 grid that is extended
  std::unique_ptr<base::Grid> fixed_dim_grid(base::Grid::createLinearGrid(base_dim));
  base::GridGenerator &fixed_grid_generator = fixed_dim_grid->getGenerator();
  fixed_grid_generator.regular(base_dim_level);
  base::GridStorage &fixed_grid_storage = fixed_dim_grid->getStorage();
  std::cout << "Fixed dim grid created! dim: " << base_dim_level
            << ", number of grid points: " << fixed_dim_grid->getSize() << std::endl;

  // read dataset
  std::cout << "reading dataset...";
  datadriven::Dataset dataset = datadriven::ARFFTools::readARFF(datasetFileName);
  std::cout << "done" << std::endl;

  std::chrono::time_point<std::chrono::system_clock> total_timer_start_without_disk =
      std::chrono::system_clock::now();

  size_t dimension = dataset.getDimension();
  std::cout << "dimension: " << dimension << std::endl;

  base::DataMatrix &trainingData = dataset.getData();
  std::cout << "data points: " << trainingData.getNrows() << std::endl;

  // create grid
  std::unique_ptr<base::Grid> grid(base::Grid::createLinearGrid(dimension));
  base::GridStorage &grid_storage = grid->getStorage();
  for (size_t j = 0; j < fixed_grid_storage.getSize(); j++) {
    base::GridPoint &p = fixed_grid_storage[j];
    base::HashGridPoint::level_type l;
    base::HashGridPoint::index_type i;
    base::GridPoint p_extended(dimension);
    for (size_t d = 0; d < base_dim; d++) {
      p.get(d, l, i);
      p_extended.set(d, l, i);
    }
    for (size_t d = base_dim; d < dimension; d++) {
      base::HashGridPoint::level_type l = 1;
      base::HashGridPoint::index_type i = 1;
      p_extended.set(d, l, i);
    }
    grid_storage.insert(p_extended);
  }
  // for (size_t j = 0; j < grid_storage.getSize(); j++) {
  //   base::GridPoint &p = grid_storage[j];
  //   std::cout << "l: ";
  //   for (size_t d = 0; d < dimension; d++) {
  //     if (d > 0) {
  //       std::cout << ", ";
  //     }
  //     base::HashGridPoint::level_type l;
  //     base::HashGridPoint::index_type i;
  //     p.get(d, l, i);
  //     std::cout << l;
  //   }
  //   std::cout << " i: ";
  //   for (size_t d = 0; d < dimension; d++) {
  //     if (d > 0) {
  //       std::cout << ", ";
  //     }
  //     base::HashGridPoint::level_type l;
  //     base::HashGridPoint::index_type i;
  //     p.get(d, l, i);
  //     std::cout << i;
  //   }
  //   std::cout << std::endl;
  // }

  base::GridGenerator &grid_generator = grid->getGenerator();
  // grid_generator.regular(level);
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

    double last_duration_generate_b = operation_mult->getLastDurationB();
    std::cout << "last_duration_generate_b: " << last_duration_generate_b << "s" << std::endl;
    double ops_generate_b =
        static_cast<double>(grid->getSize() * trainingData.getNrows() * (6 * dimension + 1)) * 1E-9;
    std::cout << "ops_generate_b: " << ops_generate_b << std::endl;
    double flops_generate_b = ops_generate_b / last_duration_generate_b;
    std::cout << "flops_generate_b: " << flops_generate_b << " GFLOPS" << std::endl;

    std::cout << "Solving density SLE" << std::endl;
    solver->solve(*operation_mult, alpha, b, false, true);

    double acc_duration_density = operation_mult->getAccDurationDensityMult();
    std::cout << "acc_duration_density: " << acc_duration_density << "s" << std::endl;

    size_t iterations = solver->getNumberIterations();
    size_t act_it = iterations + 1 + (iterations / 50);
    std::cout << "act_it: " << act_it << std::endl;
    double ops_density =
        static_cast<double>(std::pow(grid->getSize(), 2) * act_it * (14 * dimension + 2)) * 1E-9;
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

      double last_duration_generate_b = operation_mult->getLastDurationB();
      std::cout << "last_duration_generate_b: " << last_duration_generate_b << "s" << std::endl;
      double ops_generate_b =
          static_cast<double>(grid->getSize() * trainingData.getNrows() * (6 * dimension + 1)) *
          1E-9;
      std::cout << "ops_generate_b: " << ops_generate_b << std::endl;
      double flops_generate_b = ops_generate_b / last_duration_generate_b;
      std::cout << "flops_generate_b: " << flops_generate_b << " GFLOPS" << std::endl;

      std::cout << "Solving density SLE" << std::endl;
      solver->solve(*operation_mult, alpha, b, false, true);

      std::cout << "Grid points after refinement step: " << grid->getSize() << std::endl;

      size_t iterations = solver->getNumberIterations();
      size_t act_it = iterations + 1 + (iterations / 50);
      std::cout << "act_it: " << act_it << std::endl;

      double acc_duration_density = operation_mult->getAccDurationDensityMult();
      std::cout << "acc_duration_density: " << acc_duration_density << "s" << std::endl;
      double ops_density =
          static_cast<double>(std::pow(grid->getSize(), 2) * act_it * (14 * dimension + 2)) * 1E-9;
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

      double last_duration_generate_b = operation_mult->getLastDurationB();
      std::cout << "last_duration_generate_b: " << last_duration_generate_b << "s" << std::endl;
      double ops_generate_b =
          static_cast<double>(grid->getSize() * trainingData.getNrows() * (6 * dimension + 1)) *
          1E-9;
      std::cout << "ops_generate_b: " << ops_generate_b << std::endl;
      double flops_generate_b = ops_generate_b / last_duration_generate_b;
      std::cout << "flops_generate_b: " << flops_generate_b << " GFLOPS" << std::endl;

      std::cout << "Solving density SLE" << std::endl;
      solver->solve(*operation_mult, alpha, b, false, true);

      std::cout << "Grid points after coarsening step: " << grid->getSize() << std::endl;

      size_t iterations = solver->getNumberIterations();
      size_t act_it = iterations + 1 + (iterations / 50);
      std::cout << "act_it: " << act_it << std::endl;

      double acc_duration_density = operation_mult->getAccDurationDensityMult();
      std::cout << "acc_duration_density: " << acc_duration_density << "s" << std::endl;

      double ops_density =
          static_cast<double>(std::pow(grid->getSize(), 2) * act_it * (14 * dimension + 2)) * 1E-9;

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

  {
    auto total_timer_stop = std::chrono::system_clock::now();
    std::chrono::duration<double> total_elapsed_seconds = total_timer_stop - total_timer_start;
    std::chrono::duration<double> total_duration_without_disk_elapsed_seconds =
        total_timer_stop - total_timer_start_without_disk;
    double total_duration = total_elapsed_seconds.count();
    double total_duration_without_disk = total_duration_without_disk_elapsed_seconds.count();
    std::cout << "total_duration_without_disk: " << total_duration_without_disk << std::endl;
    std::cout << "total_duration: " << total_duration << std::endl;
  }
  std::cout << std::endl << "all done!" << std::endl;
}
