// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#include "sgpp/base/grid/generation/functors/SurplusRefinementFunctor.hpp"
#include "sgpp/base/opencl/OCLManagerMultiPlatform.hpp"
#include "sgpp/base/opencl/OCLOperationConfiguration.hpp"
#include "sgpp/base/operation/BaseOpFactory.hpp"
#include "sgpp/base/operation/hash/OperationMultipleEval.hpp"
#include "sgpp/datadriven/DatadrivenOpFactory.hpp"
#include "sgpp/datadriven/grid/support_refinement_iterative.hpp"
#include "sgpp/datadriven/operation/hash/DatadrivenOperationCommon.hpp"
#include "sgpp/datadriven/operation/hash/OperationMultipleEvalSubspaceAutoTuneTMP/OperationMultipleEvalSubspaceAutoTuneTMP.hpp"
#include "sgpp/datadriven/tools/ARFFTools.hpp"
#include <boost/program_options.hpp>
#include <cstdlib>
#include <experimental/filesystem>
#include <random>
#include <string>

int main(int argc, char **argv) {
  std::string datasetFileName;
  std::string scenarioName;
  std::string tunerName;
  uint32_t level;
  uint32_t repetitions_averaged;
  bool trans;
  bool isModLinear;
  bool useSupportRefinement;
  int64_t supportRefinementMinSupport;
  std::string file_prefix; // path and prefix of file name
  std::string additionalConfigFileName;

  boost::program_options::options_description description("Allowed options");

  description.add_options()("help", "display help")(
      "datasetFileName",
      boost::program_options::value<std::string>(&datasetFileName),
      "training data set as an arff or binary-arff file")(
      "scenarioName", boost::program_options::value<std::string>(&scenarioName),
      "used as the name of the created measurement files of the tuner")(
      "level", boost::program_options::value<uint32_t>(&level),
      "level of the sparse grid")(
      "tuner_name", boost::program_options::value<std::string>(&tunerName),
      "name of the auto tuning algorithm to be used")(
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
      "name for the current run, used when files are written")(
      "additionalConfig",
      boost::program_options::value<std::string>(&additionalConfigFileName),
      "(OpenCL) kernel configuration file");

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
  if (variables_map.count("tuner_name") == 0) {
    std::cerr << "error: option \"tuner_name\" not specified" << std::endl;
    return 1;
  }
  if (variables_map.count("additionalConfig") == 0) {
    std::cerr << "error: option \"additionalConfig\" not specified"
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

  // sgpp::base::AdaptivityConfiguration adaptConfig;
  // adaptConfig.maxLevelType_ = false;
  // adaptConfig.noPoints_ = 80;
  // adaptConfig.numRefinements_ = 0;
  // adaptConfig.percent_ = 200.0;
  // adaptConfig.threshold_ = 0.0;

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

  // sgpp::base::GridStorage &gridStorage = grid->getStorage();
  std::cout << "dimensionality:        " << gridStorage.getDimension()
            << std::endl;

  // sgpp::base::GridGenerator &gridGen = grid->getGenerator();
  // gridGen.regular(level);
  std::cout << "number of grid points: " << gridStorage.getSize() << std::endl;
  std::cout << "number of data points: " << dataset.getNumberInstances()
            << std::endl;

  sgpp::datadriven::OperationMultipleEvalConfiguration configuration(
      sgpp::datadriven::OperationMultipleEvalType::SUBSPACE,
      sgpp::datadriven::OperationMultipleEvalSubType::AUTOTUNETMP);
  // load (tuned) kernel configuration from file
  auto parameters = std::make_shared<sgpp::base::OCLOperationConfiguration>(
      additionalConfigFileName);
  configuration.setParameters(parameters);

  std::unique_ptr<sgpp::base::OperationMultipleEval> eval =
      std::unique_ptr<sgpp::base::OperationMultipleEval>(
          sgpp::op_factory::createOperationMultipleEval(*grid, trainingData,
                                                        configuration));
  auto &derived_eval =
      dynamic_cast<sgpp::datadriven::SubspaceAutoTuneTMP::
                       OperationMultipleEvalSubspaceAutoTuneTMP &>(*eval);
  // if (!randomization_enabled) {
  //   sgpp::datadriven::SubspaceAutoTuneTMP::
  //       OperationMultipleEvalSubspaceAutoTuneTMP &eval_cast =
  //           dynamic_cast<sgpp::datadriven::SubspaceAutoTuneTMP::
  //                            OperationMultipleEvalSubspaceAutoTuneTMP &>(
  //               *eval.get());
  //   eval_cast.set_randomize_parameter_values(false);
  // }

  std::cout << "number of grid points after refinement: "
            << gridStorage.getSize() << std::endl;
  std::cout << "grid set up" << std::endl;

  std::cout << "starting comparison..." << std::endl;

  std::string algorithm_to_tune("mult");
  if (trans) {
    algorithm_to_tune = "multTrans";
  }

  std::string full_scenario_prefix();

  std::ofstream scenario_file(file_prefix + scenarioName +
                              +"_Subspace_pvn_compare_" + algorithm_to_tune +
                              "_tuner_" + tunerName + "_t_" +
                              std::to_string(repetitions_averaged) + "av.csv");

  std::vector<std::string> parameter_names;
  if (!trans) {
    for (std::string &parameterName : (*parameters)["SubspaceMult"].keys()) {
      parameter_names.push_back(parameterName);
    }
  } else {
    for (std::string &parameterName :
         (*parameters)["SubspaceMultTrans"].keys()) {
      parameter_names.push_back(parameterName);
    }
  }
  for (std::string &par_name : parameter_names) {
    scenario_file << par_name << ", ";
  }
  scenario_file << "duration" << std::endl;

  if (!trans) {
    for (std::string &parameterName : (*parameters)["SubspaceMult"].keys()) {
      scenario_file << (*parameters)["SubspaceMult"][parameterName].get()
                    << ", ";
    }
    sgpp::base::DataVector alpha(gridStorage.getSize());
    for (size_t i = 0; i < alpha.getSize(); i++) {
      alpha[i] = static_cast<double>(i) + 1.0;
    }
    sgpp::base::DataVector dataSizeVectorResult(dataset.getNumberInstances());
    dataSizeVectorResult.setAll(0);
    double duration_sum = 0.0;
    for (size_t i = 0; i < repetitions_averaged + 1; i += 1) {
      derived_eval.mult(alpha, dataSizeVectorResult);
      if (i > 0) {
        duration_sum += derived_eval.getDuration();
      }
    }
    scenario_file << duration_sum / repetitions_averaged << std::endl;
  } else {
    for (std::string &parameterName :
         (*parameters)["SubspaceMultTrans"].keys()) {
      scenario_file << (*parameters)["SubspaceMultTrans"][parameterName].get()
                    << ", ";
    }
    sgpp::base::DataVector source(dataset.getNumberInstances());
    for (size_t i = 0; i < source.getSize(); i++) {
      source[i] = static_cast<double>(i) + 1.0;
    }
    sgpp::base::DataVector gridSizeVectorResult(grid->getSize());
    gridSizeVectorResult.setAll(0);
    double duration_sum = 0.0;
    for (size_t i = 0; i < repetitions_averaged + 1; i += 1) {
      derived_eval.multTranspose(source, gridSizeVectorResult);
      if (i > 0) {
        duration_sum += derived_eval.getDuration();
      }
    }
    scenario_file << duration_sum / repetitions_averaged << std::endl;
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
      bool ok = derived_eval.set_pvn_parameter_mult(
          *parameters, par_name, scenario_file, parameter_names);
      if (!ok) {
        std::cout << "no reset, skip" << std::endl;
        continue;
      }
      double duration_sum = 0.0;
      for (size_t i = 0; i < repetitions_averaged + 1; i += 1) {
        derived_eval.mult(alpha, dataSizeVectorResult);
        if (i > 0) {
          duration_sum += derived_eval.getDuration();
        }
      }
      scenario_file << duration_sum / repetitions_averaged << std::endl;
    } else {
      sgpp::base::DataVector source(dataset.getNumberInstances());
      for (size_t i = 0; i < source.getSize(); i++) {
        source[i] = static_cast<double>(i) + 1.0;
      }
      sgpp::base::DataVector gridSizeVectorResult(gridStorage.getSize());
      gridSizeVectorResult.setAll(0);
      bool ok = derived_eval.set_pvn_parameter_multTranspose(
          *parameters, par_name, scenario_file, parameter_names);
      if (!ok) {
        std::cout << "no reset, skip" << std::endl;
        continue;
      }
      double duration_sum = 0.0;
      for (size_t i = 0; i < repetitions_averaged + 1; i += 1) {
        derived_eval.multTranspose(source, gridSizeVectorResult);
        if (i > 0) {
          duration_sum += derived_eval.getDuration();
        }
      }
      scenario_file << duration_sum / repetitions_averaged << std::endl;
    }
  }
}
