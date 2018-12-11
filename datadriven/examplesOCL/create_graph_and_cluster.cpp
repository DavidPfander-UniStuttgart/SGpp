// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#include <boost/program_options.hpp>

#include <chrono>
#include <experimental/filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "sgpp/base/datatypes/DataVector.hpp"
#include "sgpp/base/grid/Grid.hpp"
#include "sgpp/base/grid/GridStorage.hpp"
#include "sgpp/base/grid/generation/GridGenerator.hpp"
#include "sgpp/base/opencl/OCLOperationConfiguration.hpp"
#include "sgpp/datadriven/DatadrivenOpFactory.hpp"
#include "sgpp/datadriven/operation/hash/OperationCreateGraphOCL/OpFactory.hpp"
#include "sgpp/datadriven/operation/hash/OperationCreateGraphOCL/ConnectedComponents.hpp"
#include "sgpp/datadriven/operation/hash/OperationDensityMultiplicationAVX/OperationDensityMultiplicationAVX.hpp"
#include "sgpp/datadriven/operation/hash/OperationDensityOCLMultiPlatform/OpFactory.hpp"
#include "sgpp/datadriven/operation/hash/OperationPruneGraphOCL/OpFactory.hpp"
#include "sgpp/datadriven/tools/ARFFTools.hpp"
#include "sgpp/globaldef.hpp"
#include "sgpp/solver/sle/ConjugateGradients.hpp"

using namespace sgpp;

int main(int argc, char **argv) {
  std::string datasetFileName;
  std::string configFileName;
  uint64_t k;

  boost::program_options::options_description description("Allowed options");

  description.add_options()("help", "display help")(
      "datasetFileName", boost::program_options::value<std::string>(&datasetFileName),
      "training data set as an arff file")(
      "config", boost::program_options::value<std::string>(&configFileName),
      "OpenCL and kernel configuration file")(
      "k", boost::program_options::value<uint64_t>(&k)->default_value(5),
      "specifies number of neighbors for kNN algorithm");

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

  if (variables_map.count("config") == 0) {
    std::cerr << "error: option \"config\" not specified" << std::endl;
    return 1;
  } else {
    std::experimental::filesystem::path configFilePath(configFileName);
    if (!std::experimental::filesystem::exists(configFilePath)) {
      std::cerr << "error: config file does not exist: " << configFileName << std::endl;
      return 1;
    }

    std::cout << "OpenCL configuration file: " << configFileName << std::endl;
  }

  if (variables_map.count("k") == 0) {
    std::cerr << "error: option \"k\" not specified" << std::endl;
    return 1;
  } else {
    std::cout << "k: " << k << std::endl;
  }

  // read dataset
  std::cout << "reading dataset...";
  datadriven::Dataset dataset = datadriven::ARFFTools::readARFF(datasetFileName);
  std::cout << "done" << std::endl;
  size_t dimension = dataset.getDimension();
  std::cout << "dimension: " << dimension << std::endl;
  base::DataMatrix &trainingData = dataset.getData();
  std::cout << "data points: " << trainingData.getNrows() << std::endl;

  std::cout << "starting graph creation..." << std::endl;
  auto operation_graph =
      std::unique_ptr<sgpp::datadriven::DensityOCLMultiPlatform::OperationCreateGraphOCL>(
          sgpp::datadriven::createNearestNeighborGraphConfigured(trainingData, k, dimension,
                                                                 configFileName));
  std::vector<int> graph(trainingData.getNrows() * k, -1);
  operation_graph->create_graph(graph);

  std::cout << "finding clusters..." << std::endl;
  // TODO remove this after final conversion
  std::vector<int64_t> graph_converted(graph.begin(), graph.end());
  std::vector<int64_t> node_cluster_map;
  // sgpp::datadriven::clustering::neighborhood_list_t clusters;
  // sgpp::datadriven::clustering::find_clusters(graph_converted, k, node_cluster_map, clusters);
  std::vector<std::vector<int64_t>> clusters;
  sgpp::datadriven::clustering::connected_components(graph_converted, k, node_cluster_map, clusters);  
  std::cout << "num clusters: " << clusters.size() << std::endl;
  std::cout << std::endl << "all done!" << std::endl;
}
