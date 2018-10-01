// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#include <boost/program_options.hpp>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include "sgpp/datadriven/operation/hash/OperationNearestNeighborSampled/OperationNearestNeighborSampled.hpp"
#include "sgpp/datadriven/operation/hash/OperationCreateGraphOCL/OperationCreateGraphOCL.hpp"
#include "sgpp/datadriven/tools/ARFFTools.hpp"

int main(int argc, char **argv) {
  std::string datasetFileName;
  uint64_t k;

  std::string knn_algorithm;
  uint64_t lsh_tables;
  uint64_t lsh_hashes;
  double lsh_w;

  // only for naive kNN
  std::string configFileName;

  std::string write_knn_graph;
  std::string compare_knn_csv_file_name;

  uint64_t sampling_chunk_size;
  uint64_t sampling_randomize;

  uint64_t sampling_rand_chunk_size;

  boost::program_options::options_description description("Allowed options");

  description.add_options()("help", "display help")(
      "datasetFileName", boost::program_options::value<std::string>(&datasetFileName),
      "training data set as an arff file")(
      "k", boost::program_options::value<uint64_t>(&k)->default_value(5),
      "specifies number of neighbors for kNN algorithm")(
      "knn_algorithm",
      boost::program_options::value<std::string>(&knn_algorithm)->default_value("naive_ocl"),
      "use 'lsh_cuda' (requires liblshknn), 'lsh_ocl' (requires liblshknn), "
      "'naive_ocl', 'lsh_sampling_cuda' (requires liblshknn), "
      "'lsh_sampling_opencl' (requires liblshknn), 'naive_sampling' (requires "
      "liblshknn) or 'naive' multicore (requires liblshknn)"
      "algorithm")("lsh_tables",
                   boost::program_options::value<uint64_t>(&lsh_tables)->default_value(10),
                   "number of hash tables for lsh knn")(
      "lsh_hashes", boost::program_options::value<uint64_t>(&lsh_hashes)->default_value(10),
      "number of hash functions used by lsh knn")(
      "lsh_w", boost::program_options::value<double>(&lsh_w)->default_value(1.0),
      "number of segments for hash functions used by lsh knn")(
      "write_knn_graph", boost::program_options::value<std::string>(&write_knn_graph),
      "write the knn graph calculated to a csv-file")(
      "compare_knn_csv_file_name",
      boost::program_options::value<std::string>(&compare_knn_csv_file_name),
      "compare the knn results to a reference solution")(
      "config", boost::program_options::value<std::string>(&configFileName),
      "OpenCL and kernel configuration file")(
      "sampling_chunk_size",
      boost::program_options::value<uint64_t>(&sampling_chunk_size)->default_value(0),
      "size of the chunk for sampling of dataset for knn")(
      "sampling_randomize",
      boost::program_options::value<uint64_t>(&sampling_randomize)->default_value(1),
      "how often the dataset is shuffled when using sampling knn")(
      "sampling_rand_chunk_size",
      boost::program_options::value<uint64_t>(&sampling_rand_chunk_size)->default_value(0),
      "enable chunk-level randomization for sampling knn version by setting to "
      "> 0");

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

  if (variables_map.count("compare_knn_csv_file_name") > 0) {
    std::experimental::filesystem::path configFilePath(compare_knn_csv_file_name);
    if (!std::experimental::filesystem::exists(compare_knn_csv_file_name)) {
      std::cerr << "error: compare knn csv file does not exist: " << compare_knn_csv_file_name
                << std::endl;
      return 1;
    }
  }

  //   if (knn_algorithm.compare("lsh_cuda") == 0) {
  // #ifdef LSHKNN_WITH_CUDA
  //     std::cout << "using lsh CUDA knn" << std::endl;
  // #else
  //     std::cout << "knn algorithm requires liblshknn, but SGpp was compiled "
  //                  "without liblshknn."
  //               << std::endl;
  //     return 1;
  // #endif
  //   } else if (knn_algorithm.compare("lsh_ocl") == 0) {
  // #ifdef LSHKNN_WITH_OPENCL
  //     std::cout << "using lsh OpenCL knn" << std::endl;

  // #else
  //     std::cout << "knn algorithm requires liblshknn, but SGpp was compiled "
  //                  "without liblshknn."
  //               << std::endl;
  //     return 1;
  // #endif
  //   } else if (knn_algorithm.compare("naive") == 0) {
  // #ifdef USE_LSHKNN
  //     std::cout << "using naive multicore knn" << std::endl;
  // #else
  //     std::cout << "knn algorithm requires liblshknn, but SGpp was compiled "
  //                  "without liblshknn."
  //               << std::endl;
  //     return 1;
  // #endif
  //   } else if (knn_algorithm.compare("naive_ocl") == 0) {
  //     std::cout << "using naive ocl knn" << std::endl;
  //   } else if (knn_algorithm.compare("naive_sampling") == 0) {
  // #ifdef USE_LSHKNN
  //     std::cout << "using naive sampling knn" << std::endl;
  // #else
  //     std::cout << "knn algorithm requires liblshknn, but SGpp was compiled "
  //                  "without liblshknn."
  //               << std::endl;
  //     return 1;
  // #endif
  //   } else if (knn_algorithm.compare("lsh_sampling_cuda") == 0) {
  // #ifdef LSHKNN_WITH_CUDA
  //     std::cout << "using lsh sampling CUDA knn" << std::endl;
  // #else
  //     std::cout << "knn algorithm requires liblshknn, but SGpp was compiled "
  //                  "without liblshknn."
  //               << std::endl;
  //     return 1;
  // #endif
  //   } else if (knn_algorithm.compare("lsh_sampling_ocl") == 0) {
  // #ifdef LSHKNN_WITH_OPENCL
  //     std::cout << "using lsh sampling OpenCL knn" << std::endl;
  // #else
  //     std::cout << "knn algorithm requires liblshknn, but SGpp was compiled "
  //                  "without liblshknn."
  //               << std::endl;
  //     return 1;
  // #endif
  //   } else {
  //     std::cerr << "error: invalid choice for \"knn_algorithm\" supplied"
  //               << std::endl;
  //   }

  std::cout << "reading dataset...";
  sgpp::datadriven::Dataset dataset = sgpp::datadriven::ARFFTools::readARFF(datasetFileName);
  std::cout << "done" << std::endl;

  size_t dimension = dataset.getDimension();
  std::cout << "dimension: " << dimension << std::endl;

  sgpp::base::DataMatrix &trainingData = dataset.getData();
  std::cout << "data points: " << trainingData.getNrows() << std::endl;

  if (sampling_chunk_size == 0) {
    sampling_chunk_size = trainingData.getNrows();
  }

  sgpp::datadriven::OperationNearestNeighborSampled knn_op(true);
  std::vector<int64_t> graph;
  std::chrono::time_point<std::chrono::system_clock> total_timer_start =
      std::chrono::system_clock::now();
  if (knn_algorithm.compare("lsh_cuda") == 0) {
#ifdef LSHKNN_WITH_CUDA  // purely for the compiler - program exits earlier
                         // anyway if
                         // this is not the case
    graph = knn_op.knn_lsh_cuda(dimension, trainingData, k, lsh_tables, lsh_hashes, lsh_w);
#else
    std::cerr << "error: knn_algorithm requires liblshknn with CUDA support" << std::endl;
    return 1;
#endif
  } else if (knn_algorithm.compare("lsh_ocl") == 0) {
#ifdef LSHKNN_WITH_OPENCL  // purely for the compiler - program exits earlier
                           // anyway if
                           // this is not the case
    graph = knn_op.knn_lsh_opencl(dimension, trainingData, k, configFileName, lsh_tables,
                                  lsh_hashes, lsh_w);
#else
    std::cerr << "error: knn_algorithm requires liblshknn with OpenCL support" << std::endl;
    return 1;
#endif
  } else if (knn_algorithm.compare("naive_ocl") == 0) {
    graph = knn_op.knn_naive_ocl(dimension, trainingData, k, configFileName);
  } else if (knn_algorithm.compare("naive") == 0) {
    graph = knn_op.knn_naive(dimension, trainingData, k);
  } else if (knn_algorithm.compare("naive_sampling") == 0) {
    graph = knn_op.knn_naive_sampling(dimension, trainingData, k, sampling_chunk_size,
                                      sampling_randomize);
  } else if (knn_algorithm.compare("lsh_sampling_cuda") == 0) {
#ifdef LSHKNN_WITH_CUDA  // purely for the compiler - program exits earlier
                         // anyway if
                         // this is not the case
    graph = knn_op.knn_lsh_cuda_sampling(dimension, trainingData, k, sampling_chunk_size,
                                         sampling_randomize, lsh_tables, lsh_hashes, lsh_w);
#else
    std::cerr << "error: knn_algorithm requires liblshknn with CUDA support" << std::endl;
    return 1;
#endif
  } else if (knn_algorithm.compare("lsh_sampling_opencl") == 0) {
#ifdef LSHKNN_WITH_OPENCL  // purely for the compiler - program exits earlier
                           // anyway if
                           // this is not the case
    graph = knn_op.knn_lsh_opencl_sampling(dimension, trainingData, k, configFileName,
                                           sampling_chunk_size, sampling_randomize, lsh_tables,
                                           lsh_hashes, lsh_w);
#else
    std::cerr << "error: knn_algorithm requires liblshknn with OpenCL support" << std::endl;
    return 1;
#endif
  }

  {
    std::chrono::time_point<std::chrono::system_clock> total_timer_stop =
        std::chrono::system_clock::now();
    std::chrono::duration<double> temp = total_timer_stop - total_timer_start;
    double duration_s = temp.count();
    std::cout << "duration (s): " << duration_s << std::endl;
  }

  if (variables_map.count("write_knn_graph")) {
    std::cout << "writing knn graph to file: " << write_knn_graph << "...";
    std::vector<int64_t> graph_converted(graph.begin(), graph.end());
    knn_op.write_graph_file(write_knn_graph, graph_converted, k);

    std::cout << "done" << std::endl;
  }

  if (variables_map.count("compare_knn_csv_file_name") > 0) {
    std::cout << "comparing knn result to reference result from " << compare_knn_csv_file_name
              << std::endl;

    std::vector<int64_t> neighbors_reference = knn_op.read_csv(compare_knn_csv_file_name);

    std::vector<int64_t> graph_converted(graph.begin(), graph.end());
    double acc_assigned =
        knn_op.test_accuracy(neighbors_reference, graph_converted, trainingData.getNrows(), k);
    double acc_distance = knn_op.test_distance_accuracy(
        trainingData, neighbors_reference, graph_converted, trainingData.getNrows(), dimension, k);
    std::cout << "knn correctly assigned: " << acc_assigned << std::endl;
    std::cout << "knn distance error: " << acc_distance << std::endl;
  }

}
