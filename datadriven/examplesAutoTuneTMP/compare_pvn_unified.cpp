// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

// #include "sgpp/base/grid/generation/functors/SurplusRefinementFunctor.hpp"
#include "sgpp/base/opencl/OCLManagerMultiPlatform.hpp"
#include "sgpp/base/opencl/OCLOperationConfiguration.hpp"
#include "sgpp/base/operation/BaseOpFactory.hpp"
#include "sgpp/base/operation/hash/OperationMultipleEval.hpp"
#include "sgpp/datadriven/DatadrivenOpFactory.hpp"
#include "sgpp/datadriven/grid/support_refinement_iterative.hpp"
#include "sgpp/datadriven/operation/hash/DatadrivenOperationCommon.hpp"
#include "sgpp/datadriven/operation/hash/OperationMultipleEvalStreamingModOCLUnifiedAutoTuneTMP/OperationMultiEvalStreamingModOCLUnifiedAutoTuneTMP.hpp"
#include "sgpp/datadriven/tools/ARFFTools.hpp"
#include <boost/program_options.hpp>
#include <chrono>
#include <cstdlib>
#include <experimental/filesystem>
#include <random>
#include <string>

int main(int argc, char **argv) {
  std::string datasetFileName;
  std::string OpenCLConfigFile;
  std::string scenarioName;
  uint32_t level;
  uint32_t repetitions_averaged;
  bool trans;
  bool isModLinear;
  bool useSupportRefinement;
  int64_t supportRefinementMinSupport;
  std::string file_prefix; // path and prefix of file name

  boost::program_options::options_description description("Allowed options");

  description.add_options()("help", "display help")(
      "OpenCLConfigFile",
      boost::program_options::value<std::string>(&OpenCLConfigFile),
      "the file name of the OpenCL configuration file (also used for support "
      "refinement)")(
      "datasetFileName",
      boost::program_options::value<std::string>(&datasetFileName),
      "training data set as an arff or binary-arff file")(
      "scenarioName", boost::program_options::value<std::string>(&scenarioName),
      "used as the name of the created measurement files of the tuner")(
      "level", boost::program_options::value<uint32_t>(&level),
      "level of the sparse grid")(
      "repetitions_averaged",
      boost::program_options::value<uint32_t>(&repetitions_averaged)
          ->default_value(1),
      "how many kernel calls for averaging")(
      "trans",
      boost::program_options::value<bool>(&trans)->default_value(false),
      "tune the transposed mult kernel instead of the standard one")(
      "isModLinear",
      boost::program_options::value<bool>(&isModLinear)->default_value(true),
      "use linear or mod-linear grid")(
      "use_support_refinement",
      boost::program_options::bool_switch(&useSupportRefinement),
      "use support refinement to guess an initial grid "
      "without using any solver")(
      "support_refinement_min_support",
      boost::program_options::value<int64_t>(&supportRefinementMinSupport)
          ->default_value(1),
      "for support refinement, minimal number of data points "
      "on support for accepting data point")(
      "file_prefix", boost::program_options::value<std::string>(&file_prefix),
      "name for the current run, used when files are written");

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
  if (variables_map.count("OpenCLConfigFile") == 0) {
    std::cerr << "error: option \"OpenCLConfigFile\" not specified"
              << std::endl;
    return 1;
  } else {
    std::experimental::filesystem::path filePath(OpenCLConfigFile);
    if (!std::experimental::filesystem::exists(filePath)) {
      std::cerr << "error: OpenCL configuration file does not exist: "
                << OpenCLConfigFile << std::endl;
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
  if (variables_map.count("repetitions_averaged") == 0) {
    std::cerr << "error: option \"repetitions_averaged\" not specified"
              << std::endl;
    return 1;
  }

  std::string hostname;
  if (const char *hostname_ptr = std::getenv("HOSTNAME")) {
    hostname = hostname_ptr;
    std::cout << "hostname: " << hostname << std::endl;
  } else {
    std::cerr << "error: could not query hostname from environment"
              << std::endl;
    return 1;
  }

  std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
      std::make_shared<sgpp::base::OCLOperationConfiguration>(OpenCLConfigFile);

  sgpp::datadriven::ARFFTools arffTools;
  sgpp::datadriven::Dataset dataset = arffTools.readARFF(datasetFileName);

  sgpp::base::DataMatrix &trainingData = dataset.getData();

  size_t dim = dataset.getDimension();

  std::unique_ptr<sgpp::base::Grid> grid(nullptr);
  if (isModLinear) {
    grid = std::unique_ptr<sgpp::base::Grid>(
        sgpp::base::Grid::createModLinearGrid(dim));
  } else {
    grid = std::unique_ptr<sgpp::base::Grid>(
        sgpp::base::Grid::createLinearGrid(dim));
  }

  sgpp::base::GridStorage &gridStorage = grid->getStorage();
  if (useSupportRefinement) {
    sgpp::datadriven::support_refinement_iterative ref(
        dim, level, supportRefinementMinSupport, trainingData);
    ref.enable_OCL(OpenCLConfigFile);
    ref.refine();
    std::vector<int64_t> &ls = ref.get_levels();
    if (ls.size() == 0) {
      std::cerr << "error: no grid points generated" << std::endl;
      return 1;
    }
    std::vector<int64_t> &is = ref.get_indices();
    sgpp::base::HashGridPoint p(dim);
    for (int64_t gp_index = 0; gp_index < static_cast<int64_t>(ls.size() / dim);
         gp_index += 1) {
      for (int64_t d = 0; d < static_cast<int64_t>(dim); d += 1) {
        p.set(d, ls[gp_index * dim + d], is[gp_index * dim + d]);
      }
      gridStorage.insert(p);
    }
    gridStorage.recalcLeafProperty();
    std::cout << "support refinement done, grid size: " << grid->getSize()
              << std::endl;
  } else {
    sgpp::base::GridGenerator &gridGen = grid->getGenerator();
    gridGen.regular(level);
  }

  std::cout << "dimensionality: " << dim << std::endl;
  std::cout << "number of grid points: " << gridStorage.getSize() << std::endl;
  std::cout << "number of data points: " << dataset.getNumberInstances()
            << std::endl;

  sgpp::datadriven::StreamingModOCLUnifiedAutoTuneTMP::
      OperationMultiEvalStreamingModOCLUnifiedAutoTuneTMP<double>
          eval(*grid, trainingData, parameters, isModLinear,
               repetitions_averaged);

  std::cout << "grid set up, grid size: " << grid->getSize() << std::endl;
  std::cout << "preparing operation for refined grid" << std::endl;

  eval.prepare();

  std::cout << "starting comparison..." << std::endl;
  std::string algorithm_to_tune("mult");
  if (trans) {
    algorithm_to_tune = "multTrans";
  }

  // assume there is only one device configured
  std::vector<std::string> parameter_names;
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = sgpp::datadriven::StreamingModOCLUnified::
          Configuration::getKernelName();
      json::node &kernelNode =
          deviceNode["KERNELS"].contains(kernelName)
              ? deviceNode["KERNELS"][kernelName]
              : deviceNode["KERNELS"].addDictAttr(kernelName);
      for (const std::string &par_name : kernelNode.keys()) {
        parameter_names.push_back(par_name);
      }
    }
  }

  std::ofstream scenario_file(file_prefix + scenarioName +
                              std::string("_pvn_compare_") + hostname + "_" +
                              algorithm_to_tune + "_" +
                              std::to_string(repetitions_averaged) + "av.csv");
  for (std::string &par_name : parameter_names) {
    scenario_file << par_name << ", ";
  }
  scenario_file << "duration" << std::endl;

  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = sgpp::datadriven::StreamingModOCLUnified::
          Configuration::getKernelName();
      json::node &kernelNode =
          deviceNode["KERNELS"].contains(kernelName)
              ? deviceNode["KERNELS"][kernelName]
              : deviceNode["KERNELS"].addDictAttr(kernelName);
      for (const std::string &par_name : kernelNode.keys()) {
        scenario_file << kernelNode[par_name].get() << ", ";
      }
    }
  }
  if (!trans) {
    sgpp::base::DataVector alpha(gridStorage.getSize());
    for (size_t i = 0; i < alpha.getSize(); i++) {
      alpha[i] = static_cast<double>(i) + 1.0;
    }
    sgpp::base::DataVector dataSizeVectorResult(dataset.getNumberInstances());
    dataSizeVectorResult.setAll(0);
    // repetitions handled in kernel functor
    eval.mult(alpha, dataSizeVectorResult);
    scenario_file << eval.get_last_duration_mult() << std::endl;
  } else {
    sgpp::base::DataVector source(dataset.getNumberInstances());
    for (size_t i = 0; i < source.getSize(); i++) {
      source[i] = static_cast<double>(i) + 1.0;
    }
    sgpp::base::DataVector gridSizeVectorResult(grid->getSize());
    gridSizeVectorResult.setAll(0);
    // repetitions handled in kernel functor
    eval.multTranspose(source, gridSizeVectorResult);
    scenario_file << eval.get_last_duration_multTranspose() << std::endl;
  }

  for (std::string &par_name : parameter_names) {
    std::cout << "par_name: " << par_name << std::endl;

    if (!trans) {
      sgpp::base::DataVector alpha(gridStorage.getSize());
      for (size_t i = 0; i < alpha.getSize(); i++) {
        alpha[i] = static_cast<double>(i) + 1.0;
      }
      sgpp::base::DataVector dataSizeVectorResult(dataset.getNumberInstances());
      dataSizeVectorResult.setAll(0);

      bool ok = eval.set_pvn_parameter_mult(*parameters, par_name,
                                            scenario_file, parameter_names);
      if (!ok) {
        std::cout << "no reset, skip" << std::endl;
        continue;
      }

      // repetitions handled in kernel functor
      eval.mult(alpha, dataSizeVectorResult);
      scenario_file << eval.get_last_duration_mult() << std::endl;
    } else {
      sgpp::base::DataVector source(dataset.getNumberInstances());
      for (size_t i = 0; i < source.getSize(); i++) {
        source[i] = static_cast<double>(i) + 1.0;
      }
      sgpp::base::DataVector gridSizeVectorResult(grid->getSize());
      gridSizeVectorResult.setAll(0);

      bool ok = eval.set_pvn_parameter_multTranspose(
          *parameters, par_name, scenario_file, parameter_names);
      if (!ok) {
        std::cout << "no reset, skip" << std::endl;
        continue;
      }
      // repetitions handled in kernel functor
      eval.multTranspose(source, gridSizeVectorResult);
      scenario_file << eval.get_last_duration_multTranspose() << std::endl;
    }
  }
}
