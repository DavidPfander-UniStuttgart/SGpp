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
  std::string datasetFileName = "datasets/DR5/DR5_nowarnings_less05_train.arff";
  // std::string datasetFileName = "datasets/friedman/friedman1_10d_150000.arff";
  // std::string datasetFileName = "datasets/friedman/friedman1_10d_small.arff";
  size_t level = 9;
  double lambda = 1E-2;
  // std::string configFileName = "config_ocl_float_i76700k_valgrind.cfg";
  std::string configFileName = "config_ocl_float_i76700k.cfg";

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

    // std::cout << "b: ";
    // for (size_t i = 0; i < b.size(); i++) {
    //   if (i > 0) {
    //     std::cout << ", ";
    //   }
    //   std::cout << b[i];
    // }
    // std::cout << std::endl;
    b.toFile("rhs_b.data");
  }
  std::cout << std::endl << "all done!" << std::endl;
}
