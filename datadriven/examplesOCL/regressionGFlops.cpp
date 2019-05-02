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
#include <cstdlib>
#include <experimental/filesystem>
#include <random>
#include <string>

namespace po = boost::program_options;

namespace sgpp::base {

void validate(boost::any &v, const std::vector<std::string> &values,
              sgpp::base::GridType *target_type, int) {
  // Make sure no previous assignment to 'a' was made.
  po::validators::check_first_occurrence(v);
  // Extract the first string from 'values'. If there is more than
  // one string, it's an error, and exception will be thrown.
  const std::string &s = po::validators::get_single_string(values);

  if (s.compare("Linear") == 0) {
    v = sgpp::base::GridType::Linear;
  } else if (s.compare("ModLinear") == 0) {
    v = sgpp::base::GridType::ModLinear;
  } else {
    throw po::validation_error(po::validation_error::invalid_option_value);
  }
}

} // namespace sgpp::base

namespace sgpp::datadriven {

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
  } else if (s.compare("SUBSPACE") == 0) {
    v = sgpp::datadriven::OperationMultipleEvalType::SUBSPACE;
  } else {
    throw boost::program_options::validation_error(
        boost::program_options::validation_error::invalid_option_value);
  }
}

void validate(boost::any &v, const std::vector<std::string> &values,
              sgpp::datadriven::OperationMultipleEvalSubType *target_type,
              int) {
  // Make sure no previous assignment to 'a' was made.
  po::validators::check_first_occurrence(v);
  // Extract the first string from 'values'. If there is more than
  // one string, it's an error, and exception will be thrown.
  const std::string &s = po::validators::get_single_string(values);

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
  } else if (s.compare("AUTOTUNETMP") == 0) {
    v = sgpp::datadriven::OperationMultipleEvalSubType::AUTOTUNETMP;
  } else {
    throw po::validation_error(po::validation_error::invalid_option_value);
  }
}

} // namespace sgpp::datadriven

int main(int argc, char **argv) {
  std::string datasetFileName;
  std::string OpenCLConfigFile;
  std::string scenarioName;
  std::string tunerName;
  uint32_t level;
  uint32_t repetitions_averaged;
  bool trans;
  bool isModLinear;
  bool useSupportRefinement;
  int64_t supportRefinementMinSupport;
  std::string file_prefix; // path and prefix of file name

  sgpp::datadriven::OperationMultipleEvalType type =
      sgpp::datadriven::OperationMultipleEvalType::DEFAULT;
  sgpp::datadriven::OperationMultipleEvalSubType subType =
      sgpp::datadriven::OperationMultipleEvalSubType::DEFAULT;

  po::options_description description("Allowed options");

  description.add_options()("help", "display help")(
      "OpenCLConfigFile", po::value<std::string>(&OpenCLConfigFile),
      "the file name of the OpenCL configuration file (also used for support "
      "refinement)")("datasetFileName",
                     po::value<std::string>(&datasetFileName),
                     "training data set as an arff or binary-arff file")(
      "level", po::value<uint32_t>(&level), "level of the sparse grid")(
      "repetitions_averaged", po::value<uint32_t>(&repetitions_averaged),
      "how often the performance test is repeated")(
      "trans", po::value<bool>(&trans)->default_value(false),
      "test transposed multi-eval kernel")(
      "isModLinear", po::value<bool>(&isModLinear)->default_value(true),
      "use linear or mod-linear grid")(
      "use_support_refinement", po::bool_switch(&useSupportRefinement),
      "use support refinement to guess an initial grid "
      "without using any solver")(
      "support_refinement_min_support",
      po::value<int64_t>(&supportRefinementMinSupport)->default_value(1),
      "for support refinement, minimal number of data points "
      "on support for accepting data point")(
      "operation.type",
      po::value<sgpp::datadriven::OperationMultipleEvalType>(&type),
      "implementation type of the operation")(
      "operation.subType",
      po::value<sgpp::datadriven::OperationMultipleEvalSubType>(&subType),
      "implementation sub type of the operation")
      // (
      // "file_prefix",
      // po::value<std::string>(&file_prefix), "name for the
      // current run, used when files are written")
      ;

  po::variables_map variables_map;

  po::parsed_options options = parse_command_line(argc, argv, description);
  po::store(options, variables_map);
  po::notify(variables_map);

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
  // if (variables_map.count("OpenCLConfigFile") == 0) {
  //   std::cerr << "error: option \"OpenCLConfigFile\" not specified"
  //             << std::endl;
  //   return 1;
  // } else {
  //   std::experimental::filesystem::path filePath(OpenCLConfigFile);
  //   if (!std::experimental::filesystem::exists(filePath)) {
  //     std::cerr << "error: OpenCL configuration file does not exist: "
  //               << OpenCLConfigFile << std::endl;
  //     return 1;
  //   }
  //   std::cout << "OpenCLConfigFile: " << OpenCLConfigFile << std::endl;
  // }
  if (variables_map.count("level") == 0) {
    std::cerr << "error: option \"level\" not specified" << std::endl;
    return 1;
  }
  if (variables_map.count("repetitions_averaged") == 0) {
    std::cerr << "error: option \"repetitions_averaged\" not specified" << std::endl;
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

  sgpp::base::OCLOperationConfiguration parameters;
  if (OpenCLConfigFile.compare("") != 0) {
    std::cout << "using configuration" << std::endl;
    parameters = sgpp::base::OCLOperationConfiguration(OpenCLConfigFile);
  } else {
    std::cout << "without configuration" << std::endl;
  }

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
    if (OpenCLConfigFile.compare("") != 0 && type != sgpp::datadriven::OperationMultipleEvalType::SUBSPACE) {
      ref.enable_OCL(OpenCLConfigFile);
    }
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

  // calculate number of floating point operations needed
  double num_ops = 0;
  if (!isModLinear || trans) {
    std::cout << "considering full evaluation count" << std::endl;
    num_ops += 1.0 * 1e-9 * static_cast<double>(gridStorage.getSize()) *
               static_cast<double>(dataset.getNumberInstances()) *
               static_cast<double>(dim) * 6.0 *
               static_cast<double>(repetitions_averaged);

  } else {
    std::cout << "considering reduced evaluations for l=1 grid points"
              << std::endl;
    // level 0 is skipped only if configured modlinear and only for the
    // multiEval operator
    // flops for a single non-skipped 1d hat
    double act_1d_eval_flops =
        1e-9 * 6.0 * static_cast<double>(dataset.getNumberInstances());
    int64_t no_greater_one = 0;
    for (size_t g = 0; g < grid->getSize(); g++) {
      sgpp::base::GridPoint &curPoint = grid->getStorage().getPoint(g);
      for (size_t h = 0; h < dim; h++) {
        sgpp::base::level_t level;
        sgpp::base::index_t index;
        curPoint.get(h, level, index);
        if (level > 1) {
          num_ops += act_1d_eval_flops;
          no_greater_one += 1;
        }
      }
    }
    num_ops *= static_cast<double>(repetitions_averaged);
    std::cout << "grid points with level > 1: " << no_greater_one << std::endl;
  }
  std::cout << "num_ops: " << num_ops << std::endl;

  sgpp::datadriven::OperationMultipleEvalConfiguration configuration(
      type, subType, parameters);

  std::unique_ptr<sgpp::base::OperationMultipleEval> eval(
      sgpp::op_factory::createOperationMultipleEval(*grid, trainingData,
                                                    configuration));

  std::cout << "starting performance test..." << std::endl;
  std::string algorithm_to_tune("mult");
  if (trans) {
    algorithm_to_tune = "multTrans";
  }
  std::cout << "algorithm: " << algorithm_to_tune << std::endl;

  if (!trans) {
    sgpp::base::DataVector alpha(gridStorage.getSize());
    for (size_t i = 0; i < alpha.getSize(); i++) {
      alpha[i] = static_cast<double>(i) + 1.0;
    }
    sgpp::base::DataVector dataSizeVectorResult(dataset.getNumberInstances());
    dataSizeVectorResult.setAll(0);

    // first iteration not part of timing to avoid OCL platform initialization
    // overhead problems
    eval->mult(alpha, dataSizeVectorResult);
    std::cout << "info: first mult not considered" << std::endl;

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    for (size_t i = 0; i < repetitions_averaged; i += 1) {
      eval->mult(alpha, dataSizeVectorResult);
    }
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "duration mult: " << elapsed_seconds.count()
              << ", repetitions_averaged: " << repetitions_averaged << std::endl;
    if (type == sgpp::datadriven::OperationMultipleEvalType::STREAMING) {
      std::cout << "GFLOPS: " << (num_ops / elapsed_seconds.count())
                << std::endl;
    } else {
      std::cout << "GFLOPS: " << 0 << std::endl;
    }
  } else {
    sgpp::base::DataVector source(dataset.getNumberInstances());
    for (size_t i = 0; i < source.getSize(); i++) {
      source[i] = static_cast<double>(i) + 1.0;
    }
    sgpp::base::DataVector gridSizeVectorResult(grid->getSize());
    gridSizeVectorResult.setAll(0);

    // first iteration not part of timing to avoid OCL platform initialization
    // overhead problems
    eval->multTranspose(source, gridSizeVectorResult);
    std::cout << "info: first multTranspose not considered" << std::endl;

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    for (size_t i = 0; i < repetitions_averaged; i += 1) {
      eval->multTranspose(source, gridSizeVectorResult);
    }
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "duration multTranspose: " << elapsed_seconds.count()
              << ", repetitions_averaged: " << repetitions_averaged << std::endl;
    if (type == sgpp::datadriven::OperationMultipleEvalType::STREAMING) {
      std::cout << "GFLOPS: " << (num_ops / elapsed_seconds.count())
                << std::endl;
    } else {
      std::cout << "GFLOPS: " << 0 << std::endl;
    }
  }
}
