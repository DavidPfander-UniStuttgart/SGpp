// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#include <random>
#include <string>

#include "sgpp/base/grid/generation/functors/SurplusRefinementFunctor.hpp"
#include "sgpp/base/opencl/OCLManagerMultiPlatform.hpp"
#include "sgpp/base/opencl/OCLOperationConfiguration.hpp"
#include "sgpp/base/operation/BaseOpFactory.hpp"
#include "sgpp/base/operation/hash/OperationMultipleEval.hpp"
#include "sgpp/datadriven/DatadrivenOpFactory.hpp"
#include "sgpp/datadriven/operation/hash/DatadrivenOperationCommon.hpp"
#include "sgpp/datadriven/operation/hash/OperationMultipleEvalStreamingAutoTuneTMP/OperationMultiEvalStreamingAutoTuneTMP.hpp"
#include "sgpp/datadriven/tools/ARFFTools.hpp"
#include "sgpp/globaldef.hpp"

#include "Vc/Vc"

void doAllRefinements(sgpp::base::AdpativityConfiguration& adaptConfig, sgpp::base::Grid& grid,
                      sgpp::base::GridGenerator& gridGen, sgpp::base::DataVector& alpha) {
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<double> dist(1, 100);

  for (size_t i = 0; i < adaptConfig.numRefinements_; i++) {
    sgpp::base::SurplusRefinementFunctor myRefineFunc(alpha, adaptConfig.noPoints_,
                                                      adaptConfig.threshold_);
    gridGen.refine(myRefineFunc);
    size_t oldSize = alpha.getSize();
    alpha.resize(grid.getSize());

    for (size_t j = oldSize; j < alpha.getSize(); j++) {
      alpha[j] = dist(mt);
    }
  }
}

int main(int argc, char** argv) {
  // std::string fileName = "datasets/friedman/friedman1_10d_150000.arff";
  std::string fileName = "datasets/ripley/ripleyGarcke.train.arff";

  uint32_t level = 5;

  sgpp::base::AdpativityConfiguration adaptConfig;
  adaptConfig.maxLevelType_ = false;
  adaptConfig.noPoints_ = 80;
  adaptConfig.numRefinements_ = 0;
  adaptConfig.percent_ = 200.0;
  adaptConfig.threshold_ = 0.0;

  sgpp::datadriven::ARFFTools arffTools;
  sgpp::datadriven::Dataset dataset = arffTools.readARFF(fileName);

  sgpp::base::DataMatrix& trainingData = dataset.getData();

  size_t dim = dataset.getDimension();

  bool modLinear = false;
  std::unique_ptr<sgpp::base::Grid> grid(nullptr);
  if (modLinear) {
    grid = std::unique_ptr<sgpp::base::Grid>(sgpp::base::Grid::createModLinearGrid(dim));
  } else {
    grid = std::unique_ptr<sgpp::base::Grid>(sgpp::base::Grid::createLinearGrid(dim));
  }

  sgpp::base::GridStorage& gridStorage = grid->getStorage();
  std::cout << "dimensionality:        " << gridStorage.getDimension() << std::endl;

  sgpp::base::GridGenerator& gridGen = grid->getGenerator();
  gridGen.regular(level);
  std::cout << "number of grid points: " << gridStorage.getSize() << std::endl;
  std::cout << "number of data points: " << dataset.getNumberInstances() << std::endl;

  sgpp::base::DataVector dataSizeVector(dataset.getNumberInstances());

  for (size_t i = 0; i < dataSizeVector.getSize(); i++) {
    dataSizeVector[i] = static_cast<double>(i + 1);
  }

  sgpp::datadriven::OperationMultiEvalStreamingAutoTuneTMP eval(*grid, trainingData);

  sgpp::base::DataVector alphaRefine(gridStorage.getSize());

  for (size_t i = 0; i < alphaRefine.getSize(); i++) {
    //    alpha[i] = dist(mt);
    alphaRefine[i] = static_cast<double>(i) + 1.0;
  }

  doAllRefinements(adaptConfig, *grid, gridGen, alphaRefine);

  std::cout << "number of grid points after refinement: " << gridStorage.getSize() << std::endl;
  std::cout << "grid set up" << std::endl;

  sgpp::base::DataVector alphaResult(gridStorage.getSize());
  alphaResult.setAll(0);

  std::cout << "preparing operation for refined grid" << std::endl;
  eval.prepare();

  std::cout << "calculating result" << std::endl;

  for (size_t i = 0; i < 1; i++) {
    std::cout << "repeated mult: " << i << std::endl;
    eval.multTranspose(dataSizeVector, alphaResult);
  }

  std::cout << "duration: " << eval.getDuration() << std::endl;

  std::cout << "calculating comparison values..." << std::endl;

  std::unique_ptr<sgpp::base::OperationMultipleEval> evalCompare =
      std::unique_ptr<sgpp::base::OperationMultipleEval>(
          sgpp::op_factory::createOperationMultipleEval(*grid, trainingData));

  sgpp::base::DataVector alphaResultCompare(gridStorage.getSize());

  evalCompare->multTranspose(dataSizeVector, alphaResultCompare);

  double mse = 0.0;

  double largestDifferenceMine = 0.0;
  double largestDifferenceReference = 0.0;
  double largestDifference = 0.0;

  for (size_t i = 0; i < alphaResultCompare.getSize(); i++) {
    //    std::cout << "mine: " << alphaResult[i] << " ref: " <<
    //    alphaResultCompare[i] << std::endl;
    double difference = std::abs(alphaResult[i] - alphaResultCompare[i]);
    if (difference > largestDifference) {
      largestDifference = difference;
      largestDifferenceMine = alphaResult[i];
      largestDifferenceReference = alphaResultCompare[i];
    }

    //    std::cout << "difference: " << difference << " mine: " << alphaResult[i]
    //              << " ref: " << alphaResultCompare[i] << std::endl;

    mse += difference * difference;
  }

  std::cout << "largestDifference: " << largestDifference << " mine: " << largestDifferenceMine
            << " ref: " << largestDifferenceReference << std::endl;

  mse = mse / static_cast<double>(alphaResult.getSize());
  std::cout << "mse: " << mse << std::endl;
}
