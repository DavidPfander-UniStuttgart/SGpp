// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyrigSht notice provided with SG++ or at
// sgpp.sparsegrids.org

/**
 * \page example_learnerSGDETest_cpp learner SGDE
 * This examples demonstrates density estimation.
 */

#include <random>
#include <string>

#include "sgpp/base/datatypes/DataMatrix.hpp"
#include "sgpp/base/grid/Grid.hpp"
#include "sgpp/datadriven/DatadrivenOpFactory.hpp"
#include "sgpp/datadriven/application/KernelDensityEstimator.hpp"
#include "sgpp/datadriven/application/SparseGridDensityEstimator.hpp"
#include "sgpp/datadriven/configuration/RegularizationConfiguration.hpp"
#include "sgpp/globaldef.hpp"

using sgpp::base::DataMatrix;
using sgpp::base::DataVector;
using sgpp::base::Grid;
using sgpp::base::GridGenerator;
using sgpp::base::GridStorage;

void estimate_density(double lambda) {
  /**
   * Define number of dimensions of the toy problem.
   */
  size_t numDims = 1;

  /**
   * Load normally distributed samples.
   */
  sgpp::base::DataMatrix samples(4, numDims);
  samples[0] = 0.25;
  samples[1] = 0.7;
  samples[2] = 0.75;
  samples[3] = 0.8;

  /**
   * Configure the sparse grid of level 3 with linear basis functions and the same dimension as the
   * given test data.\n
   * Alternatively load a sparse grid that has been saved to a file, see the commented line.
   */
  std::cout << "# create grid config" << std::endl;
  sgpp::base::RegularGridConfiguration gridConfig;
  gridConfig.dim_ = numDims;
  gridConfig.level_ = 4;
  gridConfig.type_ = sgpp::base::GridType::Linear;

  /**
   * Configure the adaptive refinement. Therefore the number of refinements and the number of points
   * are specified.
   */
  std::cout << "# create adaptive refinement config" << std::endl;
  sgpp::base::AdpativityConfiguration adaptConfig;
  adaptConfig.numRefinements_ = 0;
  adaptConfig.noPoints_ = 10;

  /**
   * Configure the solver. The solver type is set to the conjugent gradient method and the maximum
   * number of iterations, the tolerance epsilon and the threshold are specified.
   */
  std::cout << "# create solver config" << std::endl;
  sgpp::solver::SLESolverConfiguration solverConfig;
  solverConfig.type_ = sgpp::solver::SLESolverType::CG;
  solverConfig.maxIterations_ = 1000;
  solverConfig.eps_ = 1e-14;
  solverConfig.threshold_ = 1e-14;
  solverConfig.verbose_ = false;

  /**
   * Configure the regularization for the laplacian operator.
   */
  std::cout << "# create regularization config" << std::endl;
  sgpp::datadriven::RegularizationConfiguration regularizationConfig;
  regularizationConfig.type_ = sgpp::datadriven::RegularizationType::Laplace;

  /**
   * Configure the learner by specifying: \n
   * - ??enable,kfold?, \n
   * - an initial value for the lagrangian multiplier \f$\lambda\f$ and the interval \f$
   * [\lambda_{Start} , \lambda_{End}] \f$ in which \f$\lambda\f$ will be searched, \n
   * - whether a logarithmic scale is used, \n
   * - the parameters shuffle and an initial seed for the random value generation, \n
   * - whether parts of the output shall be kept off.
   */
  std::cout << "# create learner config" << std::endl;
  sgpp::datadriven::CrossvalidationForRegularizationConfiguration crossvalidationConfig;
  crossvalidationConfig.enable_ = false;
  crossvalidationConfig.kfold_ = 3;
  crossvalidationConfig.lambda_ = lambda;
  crossvalidationConfig.lambdaStart_ = 1e-1;
  crossvalidationConfig.lambdaEnd_ = 1e-10;
  crossvalidationConfig.lambdaSteps_ = 0;
  crossvalidationConfig.logScale_ = true;
  crossvalidationConfig.shuffle_ = true;
  crossvalidationConfig.seed_ = 1234567;
  crossvalidationConfig.silent_ = false;

  // configure learner
  std::cout << "# create learner config" << std::endl;
  sgpp::datadriven::SGDEConfiguration sgdeConfig;
  sgdeConfig.makePositive_ = false;
  // sgdeConfig.makePositive_candidateSearchAlgorithm_ =
  //     sgpp::datadriven::MakePositiveCandidateSearchAlgorithm::HybridFullIntersections;
  // sgdeConfig.makePositive_interpolationAlgorithm_ =
  //     sgpp::datadriven::MakePositiveInterpolationAlgorithm::InterpolateBoundaries1d;
  // sgdeConfig.makePositive_verbose_ = true;
  // sgdeConfig.unitIntegrand_ = true;

  /**
   * Create the learner using the configuratons set above. Then initialize it with the data read
   * from the file in the first step and train the learner.
   */
  std::cout << "# creating the learner" << std::endl;
  sgpp::datadriven::SparseGridDensityEstimator learner(gridConfig, adaptConfig, solverConfig,
                                                       regularizationConfig, crossvalidationConfig,
                                                       sgdeConfig);
  learner.initialize(samples);

  sgpp::base::Grid& grid = learner.getGrid();
  sgpp::base::DataVector& alpha = learner.getSurpluses();

  sgpp::datadriven::OperationMultipleEvalConfiguration configuration(
      sgpp::datadriven::OperationMultipleEvalType::STREAMING,
      sgpp::datadriven::OperationMultipleEvalSubType::DEFAULT);

  std::cout << "Creating multieval operation" << std::endl;
  sgpp::base::GridStorage& storage = grid.getStorage();
  sgpp::base::DataMatrix grid_point_coord(storage.getSize(), numDims);
  for (size_t i = 0; i < storage.getSize(); i++) {
    sgpp::base::HashGridPoint p = storage[i];
    // std::cout << "i: " << i << " level: " << p.getLevel(0) << " index: " << p.getIndex(0)
    //           << " coord: " << storage.getCoordinate(p, 0) << std::endl;
    grid_point_coord[i] = storage.getCoordinate(p, 0);
  }
  std::unique_ptr<sgpp::base::OperationMultipleEval> eval(
      sgpp::op_factory::createOperationMultipleEval(grid, grid_point_coord, configuration));
  sgpp::base::DataVector results(grid_point_coord.getNrows());
  std::cout << "Evaluating at evaluation grid points" << std::endl;
  eval->mult(alpha, results);

  std::cout << "x = [";
  for (size_t i = 0; i < grid_point_coord.getNrows(); i++) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << grid_point_coord[i];
  }
  std::cout << "]" << std::endl;

  std::cout << "y = [";
  for (size_t i = 0; i < results.getSize(); i++) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << results[i];
  }
  std::cout << "]" << std::endl;
}

/**
 * Now the main function begins by loading the test data from a file specified in the string
 * filename.
 */
int main(int argc, char** argv) {
  estimate_density(0.1);
  estimate_density(0.01);
  estimate_density(0.001);
}
