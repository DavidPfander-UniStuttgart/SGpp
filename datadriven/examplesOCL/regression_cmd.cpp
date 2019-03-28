// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#include <boost/program_options.hpp>
#include <chrono>
#include <string>
#include <vector>

#include "sgpp/base/opencl/manager/manager.hpp"
#include "sgpp/datadriven/application/MetaLearner.hpp"
#include "sgpp/datadriven/operation/hash/DatadrivenOperationCommon.hpp"
#include "sgpp/globaldef.hpp"
#include <sgpp/base/opencl/OCLOperationConfiguration.hpp>

namespace sgpp {
namespace base {

void validate(boost::any &v, const std::vector<std::string> &values,
              sgpp::base::GridType *target_type, int) {
  // Make sure no previous assignment to 'a' was made.
  boost::program_options::validators::check_first_occurrence(v);
  // Extract the first string from 'values'. If there is more than
  // one string, it's an error, and exception will be thrown.
  const std::string &s =
      boost::program_options::validators::get_single_string(values);

  if (s.compare("Linear") == 0) {
    v = sgpp::base::GridType::Linear;
  } else if (s.compare("ModLinear") == 0) {
    v = sgpp::base::GridType::ModLinear;
  } else {
    throw boost::program_options::validation_error(
        boost::program_options::validation_error::invalid_option_value);
  }
}
} // namespace base
} // namespace sgpp

namespace sgpp {
namespace solver {

void validate(boost::any &v, const std::vector<std::string> &values,
              sgpp::solver::SLESolverType *target_type, int) {
  // Make sure no previous assignment to 'a' was made.
  boost::program_options::validators::check_first_occurrence(v);
  // Extract the first string from 'values'. If there is more than
  // one string, it's an error, and exception will be thrown.
  const std::string &s =
      boost::program_options::validators::get_single_string(values);

  if (s.compare("CG") == 0) {
    v = sgpp::solver::SLESolverType::CG;
  } else if (s.compare("BiCGSTAB") == 0) {
    v = sgpp::solver::SLESolverType::BiCGSTAB;
  } else {
    throw boost::program_options::validation_error(
        boost::program_options::validation_error::invalid_option_value);
  }
}
} // namespace solver
} // namespace sgpp

namespace sgpp {
namespace datadriven {

void validate(boost::any &v, const std::vector<std::string> &values,
              sgpp::datadriven::OperationMultipleEvalType *target_type, int) {
  // Make sure no previous assignment to 'a' was made.
  boost::program_options::validators::check_first_occurrence(v);
  // Extract the first string from 'values'. If there is more than
  // one string, it's an error, and exception will be thrown.
  const std::string &s =
      boost::program_options::validators::get_single_string(values);

  if (s.compare("DEFAULT") == 0) {
    v = sgpp::datadriven::OperationMultipleEvalType::DEFAULT;
  } else if (s.compare("STREAMING") == 0) {
    v = sgpp::datadriven::OperationMultipleEvalType::STREAMING;
  } else if (s.compare("SUBSPACELINEAR") == 0) {
    v = sgpp::datadriven::OperationMultipleEvalType::SUBSPACELINEAR;
  } else {
    throw boost::program_options::validation_error(
        boost::program_options::validation_error::invalid_option_value);
  }
}

void validate(boost::any &v, const std::vector<std::string> &values,
              sgpp::datadriven::OperationMultipleEvalSubType *target_type,
              int) {
  // Make sure no previous assignment to 'a' was made.
  boost::program_options::validators::check_first_occurrence(v);
  // Extract the first string from 'values'. If there is more than
  // one string, it's an error, and exception will be thrown.
  const std::string &s =
      boost::program_options::validators::get_single_string(values);

  if (s.compare("DEFAULT") == 0) {
    v = sgpp::datadriven::OperationMultipleEvalSubType::DEFAULT;
  } else if (s.compare("COMBINED") == 0) {
    v = sgpp::datadriven::OperationMultipleEvalSubType::COMBINED;
  } else if (s.compare("OCL") == 0) {
    v = sgpp::datadriven::OperationMultipleEvalSubType::OCL;
  } else if (s.compare("OCLMASKMP") == 0) {
    v = sgpp::datadriven::OperationMultipleEvalSubType::OCLMASKMP;
  } else if (s.compare("OCLMP") == 0) {
    v = sgpp::datadriven::OperationMultipleEvalSubType::OCLMP;
  } else if (s.compare("SIMPLE") == 0) {
    v = sgpp::datadriven::OperationMultipleEvalSubType::SIMPLE;
  } else if (s.compare("OCLFASTMP") == 0) {
    v = sgpp::datadriven::OperationMultipleEvalSubType::OCLFASTMP;
  } else if (s.compare("OCLOPT") == 0) {
    v = sgpp::datadriven::OperationMultipleEvalSubType::OCLOPT;
  } else if (s.compare("OCLUNIFIED") == 0) {
    v = sgpp::datadriven::OperationMultipleEvalSubType::OCLUNIFIED;
  } else {
    throw boost::program_options::validation_error(
        boost::program_options::validation_error::invalid_option_value);
  }
}

enum class LearnerMode { LEARN, LEARNCOMPARE, LEARNTEST };

void validate(boost::any &v, const std::vector<std::string> &values,
              sgpp::datadriven::LearnerMode *target_type, int) {
  // Make sure no previous assignment to 'a' was made.
  boost::program_options::validators::check_first_occurrence(v);
  // Extract the first string from 'values'. If there is more than
  // one string, it's an error, and exception will be thrown.
  const std::string &s =
      boost::program_options::validators::get_single_string(values);

  if (s.compare("LEARN") == 0) {
    v = sgpp::datadriven::LearnerMode::LEARN;
  } else if (s.compare("LEARNCOMPARE") == 0) {
    v = sgpp::datadriven::LearnerMode::LEARNCOMPARE;
  } else if (s.compare("LEARNTEST") == 0) {
    v = sgpp::datadriven::LearnerMode::LEARNTEST;
  } else {
    throw boost::program_options::validation_error(
        boost::program_options::validation_error::invalid_option_value);
  }
}
} // namespace datadriven
} // namespace sgpp

int main(int argc, char *argv[]) {
  // std::string fileName = "debugging.arff";
  std::string trainingFileName = "DR5_train.arff";
  //  std::string fileName = "friedman2_90000.arff";
  //  std::string fileName = "bigger.arff";

  std::string testFileName = "";

  sgpp::datadriven::LearnerMode learnerMode =
      sgpp::datadriven::LearnerMode::LEARN;

  // only relevant for LEARNTEST-mode
  bool isRegression = true;

  double lambda = 0.000001;

  bool verbose = true;

  bool do_dummy_ocl_init = false;

  sgpp::base::RegularGridConfiguration gridConfig;
  sgpp::solver::SLESolverConfiguration SLESolverConfigRefine;
  sgpp::solver::SLESolverConfiguration SLESolverConfigFinal;
  sgpp::base::AdaptivityConfiguration adaptConfig;

  // setup grid
  gridConfig.dim_ = 0;   // dim is inferred from the data
  gridConfig.level_ = 7; // base level
  gridConfig.type_ = sgpp::base::GridType::Linear;

  // setup adaptivity
  adaptConfig.maxLevelType_ = false;
  adaptConfig.noPoints_ = 80;
  adaptConfig.numRefinements_ = 0;
  adaptConfig.percent_ = 200.0;
  adaptConfig.threshold_ = 0.0;

  // setup solver during refinement
  SLESolverConfigRefine.eps_ = 0;
  SLESolverConfigRefine.maxIterations_ = 5;
  SLESolverConfigRefine.threshold_ = -1.0;
  SLESolverConfigRefine.type_ = sgpp::solver::SLESolverType::CG;

  // setup solver for final step
  SLESolverConfigFinal.eps_ = 0;
  SLESolverConfigFinal.maxIterations_ = 5;
  SLESolverConfigFinal.threshold_ = -1.0;
  SLESolverConfigFinal.type_ = sgpp::solver::SLESolverType::CG;

  // operation type
  sgpp::datadriven::OperationMultipleEvalType type =
      sgpp::datadriven::OperationMultipleEvalType::DEFAULT;
  sgpp::datadriven::OperationMultipleEvalSubType subType =
      sgpp::datadriven::OperationMultipleEvalSubType::DEFAULT;
  // additional configuration in JSON format (primarily for OpenCL)
  std::string additionalConfigFileName;

  // parse command line options
  boost::program_options::options_description description("Allowed options");
  description.add_options()

      // general options
      ("help", "display help")(
          "trainingFileName",
          boost::program_options::value<std::string>(&trainingFileName),
          "training data set as an arff file")(
          "testFileName",
          boost::program_options::value<std::string>(&testFileName),
          "test dataset as an arff file (only for LEARNTEST-mode)")(
          "isRegression", boost::program_options::value<bool>(&isRegression),
          "true for regression, false for classification, default is "
          "regression, only relevant for LEARNTEST- and LEARN-mode")(
          "lambda", boost::program_options::value<double>(&lambda),
          "regularization parameter for learning")(
          "verbose", boost::program_options::value<bool>(&verbose),
          "do verbose learning")(
          "learnerMode",
          boost::program_options::value<sgpp::datadriven::LearnerMode>(
              &learnerMode),
          "mode of operation: LEARN -> only learning (for performance tests), "
          "LEARNTEST -> learn and use a test dataset, LEARNCOMPARE -> learn "
          "and compare with a reference implementation and compare evaluations")

      // grid setup options
      ("grid.level", boost::program_options::value<int>(&gridConfig.level_),
       "level of the initial regular grid")(
          "grid.type", boost::program_options::value<sgpp::base::GridType>(
                           &gridConfig.type_),
          "type of the grid to be used")

      // adaptivity options
      //    ("adaptivity.maxLevelType",
      //    boost::program_options::value<bool>(&adaptConfig.maxLevelType_),
      //            "DON'T KNOW WHAT THIS IS FOR")//TODO: seems to be unused,
      //            remove?
      ("adaptConfig.noPoints",
       boost::program_options::value<size_t>(&adaptConfig.noPoints_),
       "number of points to refine")(
          "adaptConfig.numRefinements",
          boost::program_options::value<size_t>(&adaptConfig.numRefinements_),
          "number of refinement steps")(
          "adaptConfig.percent",
          boost::program_options::value<double>(&adaptConfig.percent_),
          "maximum number of grid points in percent of the size of the grid "
          "that are considered for refinement")(
          "adaptConfig.threshold",
          boost::program_options::value<double>(&adaptConfig.threshold_),
          "minimum surplus value for a grid point to be considered for "
          "refinement")("adaptConfig.use_support_refinement",
                        boost::program_options::bool_switch(
                            &adaptConfig.use_support_refinement),
                        "use support refinement to guess an initial grid "
                        "without using any solver")(
          "adaptConfig.support_refinement_min_support",
          boost::program_options::value<int64_t>(
              &adaptConfig.support_refinement_min_support)
              ->default_value(1),
          "for support refinement, minimal number of data points "
          "on support for accepting data point")(
          "adaptConfig.support_refinement_ocl_config",
          boost::program_options::value<std::string>(
              &adaptConfig.support_refinement_ocl_config)
              ->default_value(""),
          "if set, uses this OCL config for creation of support-refined grid")(
          "adaptConfig.use_weight_support_coarsening",
          boost::program_options::bool_switch(
              &adaptConfig.use_weight_support_coarsening),
          "coarsen based on weighted support")(
          "adaptConfig.weight_support_coarsening_threshold",
          boost::program_options::value<double>(
              &adaptConfig.weight_support_coarsening_threshold)
              ->default_value(0.0),
          "threshold for pruning using weight support coarsening")(
          "adaptConfig.weight_support_coarsening_ocl_config",
          boost::program_options::value<std::string>(
              &adaptConfig.weight_support_coarsening_ocl_config)
              ->default_value(""),
          "weight-support requires OCL config")

      // options for the solver during refinement
      ("solverRefine.eps",
       boost::program_options::value<double>(&SLESolverConfigRefine.eps_),
       "error for early aborting training (set to 0 to disable)")(
          "solverRefine.maxIterations",
          boost::program_options::value<size_t>(
              &SLESolverConfigRefine.maxIterations_),
          "maximum number of iterations before the training is stopped")(
          "solverRefine.threshold", boost::program_options::value<double>(
                                        &SLESolverConfigRefine.threshold_),
          "early abort solver if this residual threshold is reached")(
          "solverRefine.type",
          boost::program_options::value<sgpp::solver::SLESolverType>(
              &SLESolverConfigRefine.type_),
          "the kind of solver to use")

      // options for the solver in the final step
      ("solverFinal.eps",
       boost::program_options::value<double>(&SLESolverConfigFinal.eps_),
       "error for early aborting training (set to 0 to disable)")(
          "solverFinal.maxIterations",
          boost::program_options::value<size_t>(
              &SLESolverConfigFinal.maxIterations_),
          "maximum number of iterations before the training is stopped")(
          "solverFinal.threshold", boost::program_options::value<double>(
                                       &SLESolverConfigRefine.threshold_),
          "early abort solver if this residual threshold is reached")(
          "solverFinal.type",
          boost::program_options::value<sgpp::solver::SLESolverType>(
              &SLESolverConfigFinal.type_),
          "the kind of solver to use")

      // options for the implementation type
      ("operation.type",
       boost::program_options::value<
           sgpp::datadriven::OperationMultipleEvalType>(&type),
       "implementation type of the operation")(
          "operation.subType",
          boost::program_options::value<
              sgpp::datadriven::OperationMultipleEvalSubType>(&subType),
          "implementation sub type of the operation")(
          "additionalConfig",
          boost::program_options::value<std::string>(&additionalConfigFileName),
          "(OpenCL) kernel configuration file")(
          "do_dummy_ocl_init",
          boost::program_options::bool_switch(&do_dummy_ocl_init),
          "do untimed dummy ocl init to remove duration of buggy first OCL "
          "initialization on some hardware platforms");

  boost::program_options::variables_map variables_map;

  boost::program_options::parsed_options options =
      parse_command_line(argc, argv, description);
  boost::program_options::store(options, variables_map);
  boost::program_options::notify(variables_map);

  if (variables_map.count("help")) {
    std::cout << description << std::endl;
    return 0;
  }

  std::ifstream trainingFile(trainingFileName);
  if (!trainingFile.good()) {
    std::cerr << "error: training file not found" << std::endl;
    return 1;
  }

  if (verbose) {
    std::cout << "training dataset: " << trainingFileName << std::endl;
  }

  if (learnerMode == sgpp::datadriven::LearnerMode::LEARNCOMPARE) {
    std::ifstream testFile(testFileName);
    if (!testFile.good()) {
      std::cerr << "error: test file not found" << std::endl;
      return 1;
    }
    if (verbose) {
      std::cout << "test dataset: " << testFileName << std::endl;
    }
  }

  // do dummy ocl init to not time it
  if (do_dummy_ocl_init) {
    auto total_timer_start = std::chrono::system_clock::now();
    opencl::manager_t manager(additionalConfigFileName);
    auto total_timer_stop = std::chrono::system_clock::now();
    std::chrono::duration<double> total_elapsed_seconds =
        total_timer_stop - total_timer_start;
    double total_duration = total_elapsed_seconds.count();
    std::cout << "ocl dummy initalization duration: " << total_duration
              << std::endl;
  }

  auto total_timer_start = std::chrono::system_clock::now();

  sgpp::datadriven::MetaLearner learner(gridConfig, SLESolverConfigRefine,
                                        SLESolverConfigFinal, adaptConfig,
                                        lambda, verbose);

  sgpp::datadriven::OperationMultipleEvalConfiguration configuration(type,
                                                                     subType);
  if (variables_map.count("additionalConfig")) {
    auto parameters = std::make_shared<sgpp::base::OCLOperationConfiguration>(
        additionalConfigFileName);
    configuration.setParameters(parameters);
  }

  if (learnerMode == sgpp::datadriven::LearnerMode::LEARN) {
    // only execute learning (no comparisons or tests, for performance
    // measurements)
    learner.learn(configuration, trainingFileName, isRegression);
  } else if (learnerMode == sgpp::datadriven::LearnerMode::LEARNCOMPARE) {
    // execute learning with the specified configuration and use the
    // implementation from base as comparison
    // result grids are compared by sampling the domain (again with a grid) and
    // comparing the evaluated values
    learner.learnAndCompare(configuration, trainingFileName, 8);
  } else if (learnerMode == sgpp::datadriven::LearnerMode::LEARNTEST) {
    // test the learned function with a test dataset (no cross-validation yet)
    learner.learnAndTest(configuration, trainingFileName, testFileName,
                         isRegression);
  }

  auto total_timer_stop = std::chrono::system_clock::now();
  std::chrono::duration<double> total_elapsed_seconds =
      total_timer_stop - total_timer_start;
  double total_duration = total_elapsed_seconds.count();
  std::cout << "total_duration: " << total_duration << std::endl;

  return 0;
}
