// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#include <string>
#include <utility>
#include <vector>

#include "sgpp/base/exception/factory_exception.hpp"
#include "sgpp/datadriven/DatadrivenOpFactory.hpp"
#include "sgpp/datadriven/algorithm/SystemMatrixLeastSquaresIdentity.hpp"
#include "sgpp/datadriven/application/LearnerLeastSquaresIdentity.hpp"
#include "sgpp/datadriven/tools/LearnerVectorizedPerformanceCalculator.hpp"
#include "sgpp/globaldef.hpp"

namespace sgpp {
namespace datadriven {

LearnerLeastSquaresIdentity::LearnerLeastSquaresIdentity(
    const bool isRegression, const bool verbose)
    : sgpp::datadriven::LearnerBase(isRegression, verbose) {}

// LearnerLeastSquaresIdentity::LearnerLeastSquaresIdentity(const std::string
// tGridFilename,
//                                                         const std::string
//                                                         tAlphaFilename, const
//                                                         bool isRegression,
//                                                         const bool verbose)
//    : sgpp::datadriven::LearnerBase(tGridFilename, tAlphaFilename,
//    isRegression, verbose) {}

LearnerLeastSquaresIdentity::~LearnerLeastSquaresIdentity() {}

std::unique_ptr<sgpp::datadriven::DMSystemMatrixBase>
LearnerLeastSquaresIdentity::createDMSystem(
    sgpp::base::DataMatrix &trainDataset, double lambda) {
  std::unique_ptr<sgpp::datadriven::SystemMatrixLeastSquaresIdentity>
      systemMatrix =
          std::make_unique<sgpp::datadriven::SystemMatrixLeastSquaresIdentity>(
              *(this->grid), trainDataset, lambda);
  systemMatrix->setImplementation(this->implementationConfiguration);
  return std::unique_ptr<sgpp::datadriven::DMSystemMatrixBase>(
      systemMatrix.release());
}

void LearnerLeastSquaresIdentity::postProcessing(
    const sgpp::base::DataMatrix &trainDataset,
    const sgpp::solver::SLESolverType &solver,
    const size_t numNeededIterations) {

  // iterations + right hand side + initial residual
  double actualIterations =
      static_cast<double>(numNeededIterations) + 0.5 + 1.0;
  if (reuseCoefficients) {
    // if the last alpha is reused another mult() operation is required
    actualIterations += 1.0;
  }
  // each additional recalculation of the residual after 50 iterations has the
  // cost of another iteration
  size_t additionalResidualRecalculations = numNeededIterations / 50;
  actualIterations += static_cast<double>(additionalResidualRecalculations);

  size_t nDim = grid->getDimension();
  size_t nGridsize = grid->getSize();
  size_t numInstances = trainDataset.getNrows();

  if (reuseCoefficients) {
    std::cout << "note: performance calculation: coefficients are not reused "
              << std::endl;
  } else {
    std::cout << "note: performance calculation: coefficients are reused"
              << std::endl;
  }
  std::cout << "note: performance calculation: iterations for calculation : "
            << actualIterations << std::endl;
  std::cout << "note: performance calculation: additional iteration for "
               "residual recalculation : "
            << additionalResidualRecalculations << std::endl;

  if (grid->getType() == base::GridType::Linear) {
    if (this->implementationConfiguration.getType() ==
            sgpp::datadriven::OperationMultipleEvalType::DEFAULT ||
        this->implementationConfiguration.getType() ==
            sgpp::datadriven::OperationMultipleEvalType::STREAMING) {
      if (this->implementationConfiguration.getSubType() ==
              sgpp::datadriven::OperationMultipleEvalSubType::DEFAULT ||
          this->implementationConfiguration.getSubType() ==
              sgpp::datadriven::OperationMultipleEvalSubType::OCL ||
          this->implementationConfiguration.getSubType() ==
              sgpp::datadriven::OperationMultipleEvalSubType::OCLMP ||
          this->implementationConfiguration.getSubType() ==
              sgpp::datadriven::OperationMultipleEvalSubType::OCLUNIFIED) {
        this->GFlop += 2.0 * 1e-9 * static_cast<double>(nGridsize) *
                       static_cast<double>(numInstances) *
                       static_cast<double>(nDim) * 6.0 * actualIterations;
      } else {
        std::cout << "warning: cannot calculate GFLOPS for operation subtype"
                  << std::endl;
        this->GFlop += 0;
      }
    } else {
      std::cout << "warning: cannot calculate GFLOPS for operation type"
                << std::endl;
      this->GFlop += 0;
    }
  } else if (grid->getType() == base::GridType::ModLinear) {
    if (this->implementationConfiguration.getType() ==
            sgpp::datadriven::OperationMultipleEvalType::DEFAULT ||
        this->implementationConfiguration.getType() ==
            sgpp::datadriven::OperationMultipleEvalType::STREAMING) {
      if (this->implementationConfiguration.getSubType() ==
          sgpp::datadriven::OperationMultipleEvalSubType::OCLFASTMP) {
        for (size_t g = 0; g < grid->getSize(); g++) {
          base::GridPoint &curPoint = grid->getStorage().getPoint(g);

          for (size_t h = 0; h < nDim; h++) {
            base::level_t level;
            base::index_t index;

            curPoint.get(h, level, index);

            if (level == 1) {
            } else if (index == 1) {
              this->GFlop += 1e-9 * 8.0 * actualIterations *
                             static_cast<double>(numInstances);
            } else if (index ==
                       static_cast<base::index_t>(
                           (1 << static_cast<base::index_t>(level)) - 1)) {
              this->GFlop += 1e-9 * 10.0 * actualIterations *
                             static_cast<double>(numInstances);
            } else {
              this->GFlop += 1e-9 * 12.0 * actualIterations *
                             static_cast<double>(numInstances);
            }
          }
        }
      } else if (this->implementationConfiguration.getSubType() ==
                 sgpp::datadriven::OperationMultipleEvalSubType::OCLMASKMP) {
        this->GFlop += 2.0 * 1e-9 * static_cast<double>(nGridsize) *
                       static_cast<double>(numInstances) *
                       static_cast<double>(nDim) * 6.0 * actualIterations;
      } else if (this->implementationConfiguration.getSubType() ==
                 sgpp::datadriven::OperationMultipleEvalSubType::OCLUNIFIED) {
        this->GFlop += 2.0 * 1e-9 * static_cast<double>(nGridsize) *
                       static_cast<double>(numInstances) *
                       static_cast<double>(nDim) * 6.0 * actualIterations;
      } else {
        std::cout << "warning: cannot calculate GFLOPS for operation subtype"
                  << std::endl;
        this->GFlop += 0;
      }
    } else {
      std::cout << "warning: cannot calculate GFLOPS for operation type"
                << std::endl;
      this->GFlop += 0;
    }
  } else {
    std::cout << "warning: cannot calculate GFLOPS for grid type" << std::endl;
    this->GFlop += 0;
  }
  this->GByte = 0.0;

  if (solver == solver::SLESolverType::BiCGSTAB) {
    std::cout << "warning: BiCGSTAB may be wrong (assumption: 2xFLOPS of CG)"
              << std::endl;
    this->GFlop *= 2.0;
    // result.GByte = result.GByte_ * 2.0;
  }

  // LearnerVectorizedPerformance currentPerf =
  //     LearnerVectorizedPerformanceCalculator::getGFlopAndGByte(
  //         *this->grid, trainDataset.getNrows(), solver,
  //         numNeededIterations, sizeof(double), reuseCoefficients, true);

  // this->GFlop += currentPerf.GFlop_;
  // this->GByte += currentPerf.GByte_;

  // Calculate GFLOPS and GBytes/s and write them to console
  std::cout << std::endl;
  std::cout << "Current Duration:: " << this->execTime << std::endl;
  std::cout << "Current GFlop/s: " << this->GFlop / this->execTime << std::endl;
  // std::cout << "Current GByte/s: " << this->GByte / this->execTime
  //           << std::endl;
  std::cout << std::endl;
}

void LearnerLeastSquaresIdentity::predict(
    sgpp::base::DataMatrix &testDataset,
    sgpp::base::DataVector &classesComputed) {
  // TODO: cannot reuse DMSystem because of padding -> fix!
  classesComputed.resize(testDataset.getNrows());

  sgpp::op_factory::createOperationMultipleEval(
      *(this->grid), testDataset, this->implementationConfiguration)
      ->mult(*alpha, classesComputed);
}

void LearnerLeastSquaresIdentity::multTranspose(
    sgpp::base::DataMatrix &dataset, sgpp::base::DataVector &multiplier,
    sgpp::base::DataVector &result) {
  // TODO: cannot reuse DMSystem because of padding -> fix!
  result.resize(grid->getSize());

  sgpp::op_factory::createOperationMultipleEval(
      *(this->grid), dataset, this->implementationConfiguration)
      ->mult(multiplier, result);
}

double LearnerLeastSquaresIdentity::testRegular(
    const sgpp::base::RegularGridConfiguration &GridConfig,
    sgpp::base::DataMatrix &testDataset) {
  InitializeGridRegular(GridConfig);

  std::unique_ptr<sgpp::base::OperationMultipleEval> MultEval(
      sgpp::op_factory::createOperationMultipleEval(
          *(this->grid), testDataset, this->implementationConfiguration));

  sgpp::base::DataVector classesComputed(testDataset.getNrows());

  classesComputed.setAll(0.0);

  execTime = 0.0;

  sgpp::base::SGppStopwatch *myStopwatch = new sgpp::base::SGppStopwatch();
  myStopwatch->start();

  MultEval->mult(*alpha, classesComputed);
  double stopTime = myStopwatch->stop();
  this->execTime += stopTime;
  std::cout << "execution duration: " << this->execTime << std::endl;

  return stopTime;
}

std::vector<std::pair<size_t, double>>
LearnerLeastSquaresIdentity::getRefinementExecTimes() {
  return this->ExecTimeOnStep;
}

} // namespace datadriven
} // namespace sgpp
