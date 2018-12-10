// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#include <boost/program_options.hpp>
#include <cstdlib>
#include <experimental/filesystem>
#include <random>
#include <string>
#include "sgpp/base/grid/generation/functors/SurplusRefinementFunctor.hpp"
#include "sgpp/base/opencl/OCLManagerMultiPlatform.hpp"
#include "sgpp/base/opencl/OCLOperationConfiguration.hpp"
#include "sgpp/base/operation/BaseOpFactory.hpp"
#include "sgpp/base/operation/hash/OperationMultipleEval.hpp"
#include "sgpp/datadriven/DatadrivenOpFactory.hpp"
#include "sgpp/datadriven/operation/hash/DatadrivenOperationCommon.hpp"
#include "sgpp/datadriven/operation/hash/OperationMultipleEvalStreamingOCLMultiPlatformAutoTuneTMP/OperationMultiEvalStreamingOCLMultiPlatformAutoTuneTMP.hpp"
#include "sgpp/datadriven/tools/ARFFTools.hpp"

void doAllRefinements(sgpp::base::AdaptivityConfiguration& adaptConfig, sgpp::base::Grid& grid,
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
  std::string datasetFileName;
  std::string OpenCLConfigFile;
  std::string scenarioName;
  std::string tunerName;
  uint32_t level;
  uint32_t repetitions;
  bool trans;

  boost::program_options::options_description description("Allowed options");

  description.add_options()("help", "display help")(
      "OpenCLConfigFile", boost::program_options::value<std::string>(&OpenCLConfigFile),
      "the file name of the OpenCL configuration file")(
      "datasetFileName", boost::program_options::value<std::string>(&datasetFileName),
      "training data set as an arff or binary-arff file")(
      "scenarioName", boost::program_options::value<std::string>(&scenarioName),
      "used as the name of the created measurement files of the tuner")(
      "level", boost::program_options::value<uint32_t>(&level), "level of the sparse grid")(
      "tuner_name", boost::program_options::value<std::string>(&tunerName),
      "name of the auto tuning algorithm to be used")(
      "repetitions", boost::program_options::value<uint32_t>(&repetitions),
      "how often the tuning is to be repeated")(
      "trans", boost::program_options::value<bool>(&trans)->default_value(false),
      "tune the transposed mult kernel instead of the standard one");

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
  if (variables_map.count("OpenCLConfigFile") == 0) {
    std::cerr << "error: option \"OpenCLConfigFile\" not specified" << std::endl;
    return 1;
  } else {
    std::experimental::filesystem::path filePath(OpenCLConfigFile);
    if (!std::experimental::filesystem::exists(filePath)) {
      std::cerr << "error: OpenCL configuration file does not exist: " << OpenCLConfigFile
                << std::endl;
      return 1;
    }
    std::cout << "OpenCLConfigFile: " << OpenCLConfigFile << std::endl;
  }

  if (variables_map.count("scenarioName") == 0) {
    std::cerr << "error: option \"scenarioName\" not specified" << std::endl;
    return 1;
  } else {
    std::cout << "scenarioName: " << scenarioName << std::endl;
  }
  if (variables_map.count("level") == 0) {
    std::cerr << "error: option \"level\" not specified" << std::endl;
    return 1;
  }
  if (variables_map.count("repetitions") == 0) {
    std::cerr << "error: option \"repetitions\" not specified" << std::endl;
    return 1;
  }

  std::string hostname;
  if (const char* hostname_ptr = std::getenv("HOSTNAME")) {
    hostname = hostname_ptr;
    std::cout << "hostname: " << hostname << std::endl;
  } else {
    std::cerr << "error: could not query hostname from environment" << std::endl;
    return 1;
  }

  // std::string fileName = "datasets/ripley/ripleyGarcke.train.arff";
  // std::string fileName = "datasets/friedman/friedman1_10d_150000.arff";
  // uint32_t level = 5;

  sgpp::base::AdaptivityConfiguration adaptConfig;
  adaptConfig.maxLevelType_ = false;
  adaptConfig.noPoints_ = 80;
  adaptConfig.numRefinements_ = 0;
  adaptConfig.percent_ = 200.0;
  adaptConfig.threshold_ = 0.0;

  std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
      std::make_shared<sgpp::base::OCLOperationConfiguration>(OpenCLConfigFile);

  sgpp::datadriven::ARFFTools arffTools;
  sgpp::datadriven::Dataset dataset = arffTools.readARFF(datasetFileName);

  sgpp::base::DataMatrix& trainingData = dataset.getData();

  size_t dim = dataset.getDimension();
  //    std::unique_ptr<sgpp::base::Grid> grid = sgpp::base::Grid::createLinearGrid(dim);

  // bool modLinear = false;
  std::unique_ptr<sgpp::base::Grid> grid(nullptr);
  // if (modLinear) {
  //   grid = std::unique_ptr<sgpp::base::Grid>(sgpp::base::Grid::createModLinearGrid(dim));
  // } else {
  grid = std::unique_ptr<sgpp::base::Grid>(sgpp::base::Grid::createLinearGrid(dim));
  // }

  sgpp::base::GridStorage& gridStorage = grid->getStorage();
  std::cout << "dimensionality:        " << gridStorage.getDimension() << std::endl;

  sgpp::base::GridGenerator& gridGen = grid->getGenerator();
  gridGen.regular(level);
  std::cout << "number of grid points: " << gridStorage.getSize() << std::endl;
  std::cout << "number of data points: " << dataset.getNumberInstances() << std::endl;

  sgpp::base::DataVector alpha(gridStorage.getSize());

  for (size_t i = 0; i < alpha.getSize(); i++) {
    //    alpha[i] = dist(mt);
    alpha[i] = static_cast<double>(i) + 1.0;
  }

  sgpp::datadriven::StreamingOCLMultiPlatformAutoTuneTMP::
      OperationMultiEvalStreamingOCLMultiPlatformAutoTuneTMP<double>
          eval(*grid, trainingData, parameters);

  doAllRefinements(adaptConfig, *grid, gridGen, alpha);

  std::cout << "number of grid points after refinement: " << gridStorage.getSize() << std::endl;
  std::cout << "grid set up" << std::endl;

  sgpp::base::DataVector dataSizeVectorResult(dataset.getNumberInstances());
  dataSizeVectorResult.setAll(0);

  std::cout << "preparing operation for refined grid" << std::endl;
  eval.prepare();

  std::cout << "starting tuning..." << std::endl;
  for (size_t r = 0; r < repetitions; r += 1) {
    std::cout << "rep: " << r << "(out of " << repetitions << ")" << std::endl;
    std::string algorithm_to_tune("mult");
    if (trans) {
      algorithm_to_tune = "multTrans";
    }

    std::string full_scenario_prefix(
        scenarioName + +"_" + algorithm_to_tune + "_host_" + hostname + "_tuner_" + tunerName +
        "_t_" + (*parameters)["INTERNAL_PRECISION"].get() + "_" +
        std::to_string(dataset.getNumberInstances()) + "s_" +
        std::to_string(gridStorage.getSize()) + "g_" + std::to_string(level) + "l_" +
        std::to_string(dim) + "d_" + std::to_string(r) + "r");
    if (!trans) {
      eval.tune_mult(alpha, dataSizeVectorResult, full_scenario_prefix, tunerName);
    } else {
      eval.tune_multTranspose(alpha, dataSizeVectorResult, full_scenario_prefix, tunerName);
    }
  }
}
