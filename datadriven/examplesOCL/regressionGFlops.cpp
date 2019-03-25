// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

// #include "sgpp/base/grid/generation/functors/SurplusRefinementFunctor.hpp"
#include <boost/program_options.hpp>
#include <cstdlib>
#include <experimental/filesystem>
#include <random>
#include <string>
#include "sgpp/base/opencl/OCLManagerMultiPlatform.hpp"
#include "sgpp/base/opencl/OCLOperationConfiguration.hpp"
#include "sgpp/base/operation/BaseOpFactory.hpp"
#include "sgpp/base/operation/hash/OperationMultipleEval.hpp"
#include "sgpp/datadriven/DatadrivenOpFactory.hpp"
#include "sgpp/datadriven/grid/support_refinement_iterative.hpp"
#include "sgpp/datadriven/operation/hash/DatadrivenOperationCommon.hpp"
#include "sgpp/datadriven/operation/hash/OperationMultipleEvalStreamingModOCLUnifiedAutoTuneTMP/OperationMultiEvalStreamingModOCLUnifiedAutoTuneTMP.hpp"
#include "sgpp/datadriven/tools/ARFFTools.hpp"

int main(int argc, char **argv) {
  std::string datasetFileName;
  std::string OpenCLConfigFile;
  std::string scenarioName;
  std::string tunerName;
  uint32_t level;
  uint32_t repetitions;
  bool trans;
  bool isModLinear;
  bool useSupportRefinement;
  int64_t supportRefinementMinSupport;
  std::string file_prefix;  // path and prefix of file name

  boost::program_options::options_description description("Allowed options");

  description.add_options()("help", "display help")(
      "OpenCLConfigFile", boost::program_options::value<std::string>(&OpenCLConfigFile),
      "the file name of the OpenCL configuration file (also used for support "
      "refinement)")("datasetFileName",
                     boost::program_options::value<std::string>(&datasetFileName),
                     "training data set as an arff or binary-arff file")
      // (
      // "scenarioName", boost::program_options::value<std::string>(&scenarioName),
      // "used as the name of the created measurement files of the tuner")
      ("level", boost::program_options::value<uint32_t>(&level), "level of the sparse grid")(
          "repetitions", boost::program_options::value<uint32_t>(&repetitions),
          "how often the performance test is repeated")(
          "trans", boost::program_options::value<bool>(&trans)->default_value(false),
          "test transposed multi-eval kernel")(
          "isModLinear", boost::program_options::value<bool>(&isModLinear)->default_value(true),
          "use linear or mod-linear grid")(
          "use_support_refinement", boost::program_options::bool_switch(&useSupportRefinement),
          "use support refinement to guess an initial grid "
          "without using any solver")(
          "support_refinement_min_support",
          boost::program_options::value<int64_t>(&supportRefinementMinSupport)->default_value(1),
          "for support refinement, minimal number of data points "
          "on support for accepting data point")
      // (
      // "file_prefix", boost::program_options::value<std::string>(&file_prefix),
      // "name for the current run, used when files are written")
      ;

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

  // if (variables_map.count("scenarioName") == 0) {
  //   std::cerr << "error: option \"scenarioName\" not specified" << std::endl;
  //   return 1;
  // } else {
  //   std::cout << "scenarioName: " << scenarioName << std::endl;
  // }
  if (variables_map.count("level") == 0) {
    std::cerr << "error: option \"level\" not specified" << std::endl;
    return 1;
  }
  if (variables_map.count("repetitions") == 0) {
    std::cerr << "error: option \"repetitions\" not specified" << std::endl;
    return 1;
  }

  std::string hostname;
  if (const char *hostname_ptr = std::getenv("HOSTNAME")) {
    hostname = hostname_ptr;
    std::cout << "hostname: " << hostname << std::endl;
  } else {
    std::cerr << "error: could not query hostname from environment" << std::endl;
    return 1;
  }

  sgpp::base::OCLOperationConfiguration parameters(OpenCLConfigFile);

  sgpp::datadriven::ARFFTools arffTools;
  sgpp::datadriven::Dataset dataset = arffTools.readARFF(datasetFileName);

  sgpp::base::DataMatrix &trainingData = dataset.getData();

  size_t dim = dataset.getDimension();
  std::unique_ptr<sgpp::base::Grid> grid(nullptr);
  if (isModLinear) {
    grid = std::unique_ptr<sgpp::base::Grid>(sgpp::base::Grid::createModLinearGrid(dim));
  } else {
    grid = std::unique_ptr<sgpp::base::Grid>(sgpp::base::Grid::createLinearGrid(dim));
  }

  sgpp::base::GridStorage &gridStorage = grid->getStorage();
  if (useSupportRefinement) {
    sgpp::datadriven::support_refinement_iterative ref(dim, level, supportRefinementMinSupport,
                                                       trainingData);
    ref.enable_OCL(OpenCLConfigFile);
    ref.refine();
    std::vector<int64_t> &ls = ref.get_levels();
    if (ls.size() == 0) {
      std::cerr << "error: no grid points generated" << std::endl;
      return 1;
    }
    std::vector<int64_t> &is = ref.get_indices();

    sgpp::base::HashGridPoint p(dim);
    for (int64_t gp_index = 0; gp_index < static_cast<int64_t>(ls.size() / dim); gp_index += 1) {
      for (int64_t d = 0; d < static_cast<int64_t>(dim); d += 1) {
        p.set(d, ls[gp_index * dim + d], is[gp_index * dim + d]);
      }
      gridStorage.insert(p);
    }
    gridStorage.recalcLeafProperty();
    std::cout << "support refinement done, grid size: " << grid->getSize() << std::endl;
  } else {
    sgpp::base::GridGenerator &gridGen = grid->getGenerator();
    gridGen.regular(level);
  }

  std::cout << "dimensionality: " << dim << std::endl;

  std::cout << "number of grid points: " << gridStorage.getSize() << std::endl;
  std::cout << "number of data points: " << dataset.getNumberInstances() << std::endl;

  // calculate number of floating point operations needed
  double num_ops = 0;
  if (!isModLinear || trans) {
    num_ops += 1.0 * 1e-9 * static_cast<double>(gridStorage.getSize()) *
               static_cast<double>(dataset.getNumberInstances()) * static_cast<double>(dim) * 6.0 *
               static_cast<double>(repetitions);
  } else {
    // level 0 is skipped only if configured modlinear and only for the
    // multiEval operator
    // flops for a single non-skipped 1d hat
    double act_1d_eval_flops = 1e-9 * 6.0 * static_cast<double>(dataset.getNumberInstances());
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
    num_ops *= static_cast<double>(repetitions);
    std::cout << "grid points with level > 1: " << no_greater_one << std::endl;
  }
  std::cout << "num_ops: " << num_ops << std::endl;

  sgpp::datadriven::OperationMultipleEvalConfiguration configuration(
      sgpp::datadriven::OperationMultipleEvalType::STREAMING,
      sgpp::datadriven::OperationMultipleEvalSubType::OCLUNIFIED, parameters);

  std::unique_ptr<sgpp::base::OperationMultipleEval> eval(
      sgpp::op_factory::createOperationMultipleEval(*grid, trainingData, configuration));

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
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    for (size_t i = 0; i < repetitions; i += 1) {
      eval->mult(alpha, dataSizeVectorResult);
    }
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "duration mult: " << elapsed_seconds.count() << ", repetitions: " << repetitions
              << std::endl;
    std::cout << "GFLOPS: " << (num_ops / elapsed_seconds.count()) << std::endl;
  } else {
    sgpp::base::DataVector source(dataset.getNumberInstances());
    for (size_t i = 0; i < source.getSize(); i++) {
      source[i] = static_cast<double>(i) + 1.0;
    }
    sgpp::base::DataVector gridSizeVectorResult(grid->getSize());
    gridSizeVectorResult.setAll(0);
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    for (size_t i = 0; i < repetitions; i += 1) {
      eval->multTranspose(source, gridSizeVectorResult);
    }
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "duration multTranspose: " << elapsed_seconds.count()
              << ", repetitions: " << repetitions << std::endl;
    std::cout << "GFLOPS: " << (num_ops / elapsed_seconds.count()) << std::endl;
  }
}
