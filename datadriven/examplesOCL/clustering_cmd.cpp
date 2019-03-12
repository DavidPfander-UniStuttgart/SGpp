// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#include <boost/program_options.hpp>

#include <algorithm>
#include <chrono>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#ifdef USE_LSHKNN
#include "KNNFactory.hpp"
#endif
#include "sgpp/base/datatypes/DataVector.hpp"
#include "sgpp/base/grid/Grid.hpp"
#include "sgpp/base/grid/GridStorage.hpp"
#include "sgpp/base/grid/generation/GridGenerator.hpp"
#include "sgpp/base/grid/generation/functors/SurplusCoarseningFunctor.hpp"
#include "sgpp/base/grid/generation/functors/SurplusRefinementFunctor.hpp"
#include "sgpp/base/opencl/OCLOperationConfiguration.hpp"
#include "sgpp/datadriven/DatadrivenOpFactory.hpp"
#include "sgpp/datadriven/operation/hash/OperationCreateGraphOCL/ConnectedComponents.hpp"
#include "sgpp/datadriven/operation/hash/OperationCreateGraphOCL/DetectComponents.hpp"
#include "sgpp/datadriven/operation/hash/OperationCreateGraphOCL/OpFactory.hpp"
#include "sgpp/datadriven/operation/hash/OperationDensityMultiplicationAVX/OperationDensityMultiplicationAVX.hpp"
#include "sgpp/datadriven/operation/hash/OperationDensityOCLMultiPlatform/OpFactory.hpp"
#include "sgpp/datadriven/operation/hash/OperationNearestNeighborSampled/OperationNearestNeighborSampled.hpp"
#include "sgpp/datadriven/operation/hash/OperationPruneGraphOCL/OpFactory.hpp"
#include "sgpp/datadriven/tools/ARFFTools.hpp"
#include "sgpp/globaldef.hpp"
#include "sgpp/solver/sle/ConjugateGradients.hpp"

#include "sgpp/datadriven/grid/spatial_refinement_blocked.hpp"

#include <sgpp/base/grid/generation/functors/WeightSupportCoarseningFunctor.hpp>
// #include <sgpp/base/grid/generation/functors/UnlimitedCoarseningFunctor.hpp>
// namespace sgpp {
// namespace base {
// class DensityBCoarseningFunctor : public UnlimitedCoarseningFunctor {
// protected:
//   DataVector &b;
//   double threshold;

// public:
//   /**
//    * Constructor.
//    *
//    * @param alpha DataVector that is basis for coarsening decisions. The i-th
//    * entry corresponds to the i-th grid point.
//    * @param threshold The absolute value of the entries have to be less or
//    equal
//    * than the threshold to be considered for coarsening
//    */
//   DensityBCoarseningFunctor(DataVector &b, double threshold = 0.0)
//       : b(b), threshold(threshold) {}

//   ~DensityBCoarseningFunctor() override {}

//   bool operator()(GridStorage &storage, size_t seq) override {
//     return b[seq] < threshold; // sums of hat functions are positive
//   }

//   // double start() const override { return 1.0; }

//   // size_t getRemovementsNum() const override {
//   //   // interface does not allow for infinity
//   //   // TODO: stupid implementation, buffer of that size is allocated
//   //   return 500000;
//   // }

//   // double getCoarseningThreshold() const override { return this->threshold;
//   }
// };

// } // namespace base
// } // namespace sgpp

// namespace util {
// std::vector<int64_t> read_vector(const std::string &file_name) {
//   std::vector<int64_t> v;
//   std::ifstream ifs(file_name, std::ios::in | std::ifstream::binary);
//   std::istream_iterator<int64_t> iter(ifs);
//   std::istream_iterator<int64_t> end;
//   std::copy(iter, end, std::back_inserter(v));
//   return v;
// }
// } // namespace util

double testAccuracy(const std::vector<int> correct,
                    const std::vector<int> result, const int size,
                    const int k) {
  std::cout << "size: " << size << std::endl;
  int count = 0;
  int total = 0;
  for (int s = 0; s < size; s += 1) {
    for (int c = 0; c < k; c += 1) {
      for (int r = 0; r < k; r += 1) {
        // if (correct[s * k + c] < 0 && result[s * k + r] < 0) {
        //   total += 1;
        //   continue;
        // }
        if (correct[s * k + c] == result[s * k + r]) {
          count += 1;
          break;
        }
      }
      total += 1;
    }
  }
  std::cout << "count: " << count << std::endl;
  std::cout << "total: " << total << std::endl;
  return static_cast<double>(count) / static_cast<double>(total);
}

double testDistanceAccuracy(const std::vector<double> data,
                            const std::vector<int> correct,
                            const std::vector<int> result, const int size,
                            const int dim, const int k) {
  double dist_sum_correct = 0.0;
  double dist_sum_lsh = 0.0;
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < k; ++j) {
      // if (correct[k * i + j] < 0) {
      //   continue;
      // }

      double dist_correct = 0.0;
      for (int d = 0; d < dim; ++d) {
        dist_correct +=
            (data[size * d + i] - data[size * d + correct[k * i + j]]) *
            (data[size * d + i] - data[size * d + correct[k * i + j]]);
      }
      dist_sum_correct += sqrt(dist_correct);
    }
    for (int j = 0; j < k; ++j) {
      // if (result[k * i + j] < 0) {
      //   continue;
      // }
      double dist_lsh = 0.0;
      for (int d = 0; d < dim; ++d) {
        dist_lsh += (data[size * d + i] - data[size * d + result[k * i + j]]) *
                    (data[size * d + i] - data[size * d + result[k * i + j]]);
      }
      dist_sum_lsh += sqrt(dist_lsh);
    }
  }
  return 1.0 - std::abs(dist_sum_lsh / dist_sum_correct);
}

std::vector<std::string> split(const std::string &s, char delim) {
  std::stringstream ss(s);
  std::string item;
  std::vector<std::string> r;

  while (std::getline(ss, item, delim)) {
    r.push_back(item);
  }

  return r;
}

using namespace sgpp;

int main(int argc, char **argv) {
  std::string datasetFileName;
  std::string binary_header_filename;
  size_t level;
  double lambda;
  std::string configFileName;
  uint64_t k;
  double threshold;

  bool write_all;
  bool write_density_grid;
  size_t eval_grid_level; // for writing evaluated density function
  bool write_evaluated_density_full_grid;
  bool write_knn_graph;
  bool write_pruned_knn_graph;
  bool write_rhs;
  bool write_cluster_map;

  bool record_timings;
  int64_t print_cluster_sizes;
  std::string file_prefix; // path and prefix of file name
  int64_t target_clusters;
  std::string reuse_density_grid;
  std::string reuse_knn_graph;
  std::string compare_knn_csv_file_name;
  double epsilon;
  int64_t max_iterations;
  // int64_t connected_components_k_factor;

  // for right-hand side coarsening
  bool weight_support_coarsening;
  double weight_support_coarsening_threshold;

  bool use_support_refinement;
  int64_t support_refinement_min_support;

  size_t refinement_steps;
  size_t refinement_points;
  size_t coarsening_points;
  double coarsening_threshold;
  std::string knn_algorithm;
  uint64_t lsh_tables;
  uint64_t lsh_hashes;
  double lsh_w;

  uint64_t sampling_chunk_size;
  uint64_t sampling_chunk_count;

  boost::program_options::options_description description("Allowed options");

  description.add_options()("help", "display help")(
      "datasetFileName",
      boost::program_options::value<std::string>(&datasetFileName),
      "training data set as an arff file")(
      "binary_header_filename",
      boost::program_options::value<std::string>(&binary_header_filename),
      "training data set as an arff file")(
      "density_eval_full_grid_level",
      boost::program_options::value<size_t>(&eval_grid_level)->default_value(2),
      "level for the evaluation of the sparse grid density function on "
      "printable full grid")(
      "level", boost::program_options::value<size_t>(&level)->default_value(4),
      "level of the sparse grid used for density estimation")(
      "lambda",
      boost::program_options::value<double>(&lambda)->default_value(0.000001),
      "regularization for density estimation")(
      "config", boost::program_options::value<std::string>(&configFileName),
      "OpenCL and kernel configuration file")(
      "k", boost::program_options::value<uint64_t>(&k)->default_value(5),
      "specifies number of neighbors for kNN algorithm")(
      "reuse_knn_graph",
      boost::program_options::value<std::string>(&reuse_knn_graph),
      "name of a file containing the knn graph of the dataset; knn graph is "
      "not computed if this option is set")(
      "reuse_density_grid",
      boost::program_options::value<std::string>(&reuse_density_grid),
      "name of a file containing the sparse grid for reuse")(
      "epsilon",
      boost::program_options::value<double>(&epsilon)->default_value(0.0001),
      "Exit criteria for the solver. Usually ranges from 0.001 to 0.0001.")(
      "max_iterations",
      boost::program_options::value<int64_t>(&max_iterations)
          ->default_value(1000),
      "The maximum number of CG iterations for the density estimation solver.")(
      "threshold",
      boost::program_options::value<double>(&threshold)
          ->default_value(-99999.0),
      "threshold for sparse grid function for removing edges")(
      "file_prefix", boost::program_options::value<std::string>(&file_prefix),
      "name for the current run, used when files are written")(
      "refinement_steps",
      boost::program_options::value<uint64_t>(&refinement_steps)
          ->default_value(0),
      "number of refinment steps for density estimation")(
      "refinement_points",
      boost::program_options::value<uint64_t>(&refinement_points)
          ->default_value(0),
      "number of points to refinement during density estimation")(
      "coarsen_points",
      boost::program_options::value<uint64_t>(&coarsening_points)
          ->default_value(0),
      "number of points to coarsen during density estimation")(
      "coarsen_threshold",
      boost::program_options::value<double>(&coarsening_threshold)
          ->default_value(1000.0),
      "for density estimation, only surpluses below threshold are "
      "coarsened")(
      "knn_algorithm",
      boost::program_options::value<std::string>(&knn_algorithm)
          ->default_value("naive_ocl"),
      "type of kNN algorithm used, either 'lsh_cuda' (requires liblshknn), "
      "'lsh_ocl' (requires liblshknn), 'naive_ocl' or 'naive' (requires "
      "liblshknn)")(
      "lsh_tables",
      boost::program_options::value<uint64_t>(&lsh_tables)->default_value(50),
      "number of hash tables for lsh knn")(
      "lsh_hashes",
      boost::program_options::value<uint64_t>(&lsh_hashes)->default_value(15),
      "number of hash functions used by lsh knn")(
      "lsh_w",
      boost::program_options::value<double>(&lsh_w)->default_value(1.5),
      "number of segments for hash functions used by lsh knn")(
      "write_all", boost::program_options::bool_switch(&write_all),
      "write all results to file (does not evaluate density grid)")(
      "write_knn_graph", boost::program_options::bool_switch(&write_knn_graph),
      "write the knn graph calculated to a csv-file")(
      "write_rhs", boost::program_options::bool_switch(&write_rhs),
      "Filename where the final rhs values will be written.")(
      "write_knn", boost::program_options::bool_switch(&write_knn_graph),
      "write knn graph to file")(
      "write_pruned_knn_graph",
      boost::program_options::bool_switch(&write_pruned_knn_graph),
      "write the pruned knn graph calculated to a csv-file")(
      "write_cluster_map",
      boost::program_options::bool_switch(&write_cluster_map),
      "write mapped clusters to a csv-file")(
      "write_density_grid",
      boost::program_options::bool_switch(&write_density_grid),
      "write the coordinates, levels, indices and surpluses to a csv-file")(
      "write_evaluated_density_full_grid",
      boost::program_options::bool_switch(&write_evaluated_density_full_grid),
      "evaluate density function on full grid and write result to a "
      "csv-file")("record_timings",
                  boost::program_options::bool_switch(&record_timings),
                  "write runtime performance measurements to a csv-file")(
      "compare_knn_csv_file_name",
      boost::program_options::value<std::string>(&compare_knn_csv_file_name),
      "compare the knn results to a reference solution")(
      "sampling_chunk_size",
      boost::program_options::value<uint64_t>(&sampling_chunk_size)
          ->default_value(0),
      "size of the chunk for sampling of dataset for knn")(
      "sampling_chunk_count",
      boost::program_options::value<uint64_t>(&sampling_chunk_count)
          ->default_value(1),
      "number of chunks for sampling of dataset for knn")(
      "target_clusters",
      boost::program_options::value<int64_t>(&target_clusters)
          ->default_value(0),
      "for calculating clustering score if value > 0 is given")(
      "print_cluster_sizes",
      boost::program_options::value<int64_t>(&print_cluster_sizes)
          ->default_value(0),
      "print the cluster sizes to stdout, value is min cluster size")(
      "use_weight_support_coarsening",
      boost::program_options::bool_switch(&weight_support_coarsening),
      "for density estimation, use the vector for cheap support-based "
      "coarsening")(
      "weight_support_coarsening_threshold",
      boost::program_options::value<double>(
          &weight_support_coarsening_threshold)
          ->default_value(0.0),
      "for density estimation, prune if per grid points sum is <= given "
      "threshold")(
      "use_support_refinement",
      boost::program_options::bool_switch(&use_support_refinement),
      "use support refinement to guess an initial grid without using the CG "
      "solver")(
      "support_refinement_min_support",
      boost::program_options::value<int64_t>(&support_refinement_min_support)
          ->default_value(1),
      "for support refinement, minimal number of data points on support for "
      "accepting data "
      "point")
      // (
      // "connected_components_k_factor",
      // boost::program_options::value<int64_t>(&connected_components_k_factor)->default_value(1),
      // "add some additional entries (a factor of k) to the kNN graph to reduce
      // the number of merges " "required")
      ;

  boost::program_options::variables_map variables_map;

  boost::program_options::parsed_options options =
      parse_command_line(argc, argv, description);
  boost::program_options::store(options, variables_map);
  boost::program_options::notify(variables_map);

  if (variables_map.count("help")) {
    std::cout << description << std::endl;
    return 0;
  }

  if (write_all) {
    write_knn_graph = true;
    write_pruned_knn_graph = true;
    write_rhs = true;
    write_cluster_map = true;
    write_density_grid = true;
  }

  if (write_all || write_knn_graph || write_pruned_knn_graph || write_rhs ||
      write_cluster_map || write_density_grid) {
    if (file_prefix.compare("") == 0) {
      std::cerr << "error: cannot write results (write_*-options) without "
                   "\"file_prefix\""
                << std::endl;
      return 1;
    }
  }

  if (variables_map.count("datasetFileName") == 0 &&
      variables_map.count("binary_header_filename") == 0) {
    std::cerr << "error: neither option \"datasetFileName\" is specified "
              << std::endl;
    std::cerr << "nor is the option \"binary_header_filename\"" << std::endl;
    return 1;
  } else if (variables_map.count("datasetFileName") != 0 &&
             variables_map.count("binary_header_filename") == 0) {
    std::experimental::filesystem::path datasetFilePath(datasetFileName);
    if (!std::experimental::filesystem::exists(datasetFilePath)) {
      std::cerr << "error: dataset file does not exist: " << datasetFileName
                << std::endl;
      return 1;
    }
    std::cout << "datasetFileName: " << datasetFileName << std::endl;
  } else if (variables_map.count("datasetFileName") == 0 &&
             variables_map.count("binary_header_filename") != 0) {
    std::experimental::filesystem::path datasetFilePath(binary_header_filename);
    if (!std::experimental::filesystem::exists(datasetFilePath)) {
      std::cerr << "error: dataset file does not exist: "
                << binary_header_filename << std::endl;
      return 1;
    }
    std::cout << "binary_header_filename: " << binary_header_filename
              << std::endl;
  } else {
    std::cerr << "error: Both options \"datasetFileName\" and " << std::endl;
    std::cerr << "binary_header_filename\" are specified! Use only one. "
              << std::endl;
    return 1;
  }

  if (variables_map.count("level") == 0) {
    std::cerr << "error: option \"level\" not specified" << std::endl;
    return 1;
  } else {
    std::cout << "level: " << level << std::endl;
  }

  if (variables_map.count("lambda") == 0) {
    std::cerr << "error: option \"lambda\" not specified" << std::endl;
    return 1;
  } else {
    std::cout << "lambda: " << lambda << std::endl;
  }

  if (variables_map.count("config") == 0) {
    std::cerr << "error: option \"config\" not specified" << std::endl;
    return 1;
  } else {
    std::experimental::filesystem::path configFilePath(configFileName);
    if (!std::experimental::filesystem::exists(configFilePath)) {
      std::cerr << "error: config file does not exist: " << configFileName
                << std::endl;
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

  if (variables_map.count("threshold") == 0) {
    std::cerr << "error: option \"threshold\" not specified" << std::endl;
    return 1;
  } else {
    std::cout << "threshold: " << threshold << std::endl;
  }

  if (knn_algorithm.compare("lsh_cuda") == 0) {
#ifdef LSHKNN_WITH_CUDA
    std::cout << "using lsh CUDA knn" << std::endl;
#else
    std::cout << "knn algorithm requires liblshknn, but SGpp was compiled "
                 "without liblshknn."
              << std::endl;
    return 1;
#endif
  } else if (knn_algorithm.compare("lsh_ocl") == 0) {
    std::cout << "using lsh OpenCL knn" << std::endl;
  } else if (knn_algorithm.compare("naive") == 0) {
#ifdef USE_LSHKNN
    std::cout << "using naive multicore knn" << std::endl;
#else
    std::cout << "knn algorithm requires liblshknn, but SGpp was compiled "
                 "without liblshknn."
              << std::endl;
    return 1;
#endif
  } else if (knn_algorithm.compare("naive_ocl") == 0) {
    std::cout << "using naive ocl knn" << std::endl;
  } else {
    std::cerr << "error: invalid choice for \"knn_algorithm\" supplied"
              << std::endl;
    return 1;
  }

  std::ofstream result_timings;
  if (record_timings) {
    std::string result_timings_file_name(file_prefix + "_result_timings.csv");
    std::experimental::filesystem::path result_timings_path(
        result_timings_file_name);
    if (std::experimental::filesystem::exists(result_timings_path)) {
      result_timings.open(file_prefix + "_result_timings.csv",
                          std::ios::out | std::ios::app);
    } else {
      result_timings.open(file_prefix + "_result_timings.csv", std::ios::out);
      result_timings
          << "dataset; grid_level; lambda; threshold; k; config; "
             "refine_steps; "
             "refine_points; coarsen_points; coarsen_threshold; "
             "duration_generate_b; "
             "gflops_generate_b; duration_density_average; "
             "gflops_density_average; "
             "duration_create_graph; gflops_create_graph; "
             "duration_prune_graph; "
             "gflops_prune_graph; total_duration_without_disk;total_duration"
          << std::endl;
    }
    result_timings << datasetFileName << "; " << level << "; " << lambda << "; "
                   << threshold << "; " << k << "; " << configFileName << ";"
                   << refinement_steps << "; " << refinement_points << "; "
                   << coarsening_points << "; " << coarsening_threshold << "; ";
  }

  std::chrono::time_point<std::chrono::system_clock> total_timer_start =
      std::chrono::system_clock::now();

  // Loading dataset
  std::cout << "reading dataset..." << std::endl;
  size_t dimension = 0;

  std::chrono::time_point<std::chrono::system_clock> loading_data_start,
      loading_data_end;
  loading_data_start = std::chrono::system_clock::now();

  base::DataMatrix trainingData;
  if (variables_map.count("binary_header_filename") != 0) {
    std::cout << "info: binary dataset" << std::endl;
    trainingData = sgpp::datadriven::ARFFTools::read_binary_converted_ARFF(
        binary_header_filename);
    dimension = trainingData.getNcols();
  } else if (variables_map.count("datasetFileName") != 0) {
    std::cout << "info: ARFF dataset" << std::endl;
    sgpp::datadriven::Dataset dataset =
        sgpp::datadriven::ARFFTools::readARFF(datasetFileName);
    trainingData = std::move(dataset.getData());
    dimension = dataset.getDimension();
  } else {
    throw; // should never end up here
  }

  loading_data_end = std::chrono::system_clock::now();
  std::cout << "Dataset load duration: "
            << static_cast<std::chrono::duration<double>>(loading_data_end -
                                                          loading_data_start)
                   .count()
            << std::endl;

  std::cout << "dimension: " << dimension << std::endl;
  std::cout << "data points: " << trainingData.getNrows() << std::endl;
  if (trainingData.getNrows() == 0) {
    std::cerr << "error: dataset is empty" << std::endl;
    return 1;
  }
  // // read dataset
  std::cout << "printing first two datapoints (if available):" << std::endl;
  for (size_t i = 0; i < std::min(trainingData.getNrows(), 2ul); i += 1) {
    for (size_t d = 0; d < dimension; d += 1) {
      if (d > 0) {
        std::cout << ", ";
      }
      std::cout << trainingData[i * dimension + d];
    }
    std::cout << std::endl;
  }

  std::chrono::time_point<std::chrono::system_clock>
      total_timer_start_without_disk = std::chrono::system_clock::now();

  // create grid
  std::unique_ptr<base::Grid> grid;
  base::DataVector alpha;

  if (reuse_density_grid.compare("") != 0) {
    std::cout << "reusing density grid and coefficients" << std::endl;
    std::chrono::time_point<std::chrono::system_clock> reuse_grid_start,
        reuse_grid_stop;
    reuse_grid_start = std::chrono::system_clock::now();
    std::ifstream in_grid(reuse_density_grid);
    grid = std::unique_ptr<base::Grid>(base::Grid::unserialize(in_grid));

    std::string reuse_density_coef(reuse_density_grid);
    size_t i = reuse_density_coef.find("_density_grid.serialized");
    std::string coef_pattern("_density_coef.serialized");
    reuse_density_coef.replace(i, coef_pattern.size(), coef_pattern);
    alpha = sgpp::base::DataVector::fromFile(reuse_density_coef);
    reuse_grid_stop = std::chrono::system_clock::now();
    std::chrono::duration<double> reuse_grid_seconds =
        reuse_grid_stop - reuse_grid_start;
    std::cout << "loading density grid (reuse) duration: "
              << reuse_grid_seconds.count() << "s" << std::endl;

    // for (size_t i = 0; i < grid->getSize(); ++i) {
    //   out_coefficients << alpha[i] << std::endl;
    // }
    // out_coefficients.close();
  } else {
    std::chrono::time_point<std::chrono::system_clock> grid_create_start,
        grid_create_stop;
    grid_create_start = std::chrono::system_clock::now();
    if (!use_support_refinement) {
      grid =
          std::unique_ptr<base::Grid>(base::Grid::createLinearGrid(dimension));
      base::GridGenerator &grid_generator = grid->getGenerator();
      grid_generator.regular(level);
      std::cout << "initial grid created, grid points: " << grid->getSize()
                << std::endl;
    } else {
      std::cout << "support_refinement_min_support:"
                << support_refinement_min_support << std::endl;
      sgpp::datadriven::spatial_refinement_blocked ref(
          dimension, level, support_refinement_min_support, trainingData);
      ref.enable_OCL(configFileName);
      ref.refine();
      std::vector<int64_t> &ls = ref.get_levels();
      if (ls.size() == 0) {
        std::cerr << "error: no grid points generated" << std::endl;
        return 1;
      }
      std::vector<int64_t> &is = ref.get_indices();
      grid = std::unique_ptr<sgpp::base::Grid>(
          sgpp::base::Grid::createLinearGrid(dimension));
      sgpp::base::GridStorage &grid_storage = grid->getStorage();
      sgpp::base::HashGridPoint p(dimension);
      for (int64_t gp_index = 0;
           gp_index < static_cast<int64_t>(ls.size() / dimension);
           gp_index += 1) {
        for (int64_t d = 0; d < static_cast<int64_t>(dimension); d += 1) {
          p.set(d, ls[gp_index * dimension + d], is[gp_index * dimension + d]);
        }
        grid_storage.insert(p);
      }
      grid_storage.recalcLeafProperty();
      std::cout << "support refinement done, grid size: " << grid->getSize()
                << std::endl;
    }
    // sgpp::base::GridStorage &grid_storage = grid->getStorage();

    // for (size_t gp_index = 0; gp_index < grid_storage.getSize();
    //      gp_index += 1) {
    //   sgpp::base::HashGridPoint &p = grid_storage[gp_index];
    //   uint32_t cur_l, cur_i;
    //   std::cout << "linear index: " << gp_index << " l: ";
    //   for (int64_t d = 0; d < static_cast<int64_t>(dimension); d += 1) {
    //     p.get(d, cur_l, cur_i);
    //     if (d > 0) {
    //       std::cout << ", ";
    //     }
    //     std::cout << cur_l;
    //   }
    //   std::cout << " i: ";
    //   for (int64_t d = 0; d < static_cast<int64_t>(dimension); d += 1) {
    //     p.get(d, cur_l, cur_i);
    //     if (d > 0) {
    //       std::cout << ", ";
    //     }
    //     std::cout << cur_i;
    //   }
    //   std::cout << std::endl;
    // }

    alpha.resize(grid->getSize());
    alpha.setAll(0.0);

    grid_create_stop = std::chrono::system_clock::now();
    std::chrono::duration<double> grid_create_seconds =
        grid_create_stop - grid_create_start;
    std::cout << "grid create duration: " << grid_create_seconds.count() << "s"
              << std::endl;

    // create solver
    std::chrono::time_point<std::chrono::system_clock> density_timer_start;
    std::chrono::time_point<std::chrono::system_clock> density_timer_stop;
    density_timer_start = std::chrono::system_clock::now();
    auto solver =
        std::make_unique<solver::ConjugateGradients>(max_iterations, epsilon);

    base::DataVector b(grid->getSize(), 0.0);

    // add one to at least run the density estimation once
    refinement_steps += 1;

    base::GridGenerator &grid_generator = grid->getGenerator();

    for (size_t i = 0; i < refinement_steps; i++) {
      if (coarsening_points > 0 && i > 0) {
        size_t grid_size_before_coarsen = grid->getSize();
        sgpp::base::SurplusCoarseningFunctor coarsen_func(
            alpha, coarsening_points, coarsening_threshold);
        grid_generator.coarsen(coarsen_func, alpha);

        size_t grid_size_after_coarsen = grid->getSize();
        std::cout << "coarsen: removed "
                  << (grid_size_before_coarsen - grid_size_after_coarsen)
                  << " grid points" << std::endl;
      }
      if (refinement_points > 0 && i > 0) {
        sgpp::base::SurplusRefinementFunctor refine_func(alpha,
                                                         refinement_points);
        grid_generator.refine(refine_func);
        size_t old_size = alpha.getSize();

        // adjust alpha to refined grid
        alpha.resize(grid->getSize());
        for (size_t j = old_size; j < alpha.getSize(); j++) {
          alpha[j] = 0.0;
        }

        // regenerate b with refined grid
        b.resize(grid->getSize());
        for (size_t j = old_size; j < alpha.getSize(); j++) {
          b[j] = 0.0;
        }
        std::cout << "surplus refinement: new grid size " << grid->getSize()
                  << std::endl;
      }

      std::unique_ptr<datadriven::DensityOCLMultiPlatform::OperationDensity>
          operation_mult(datadriven::createDensityOCLMultiPlatformConfigured(
              *grid, dimension, lambda, configFileName));

      operation_mult->generateb(trainingData, b);

      double last_duration_generate_b = operation_mult->getLastDurationB();
      std::cout << "last_duration_generate_b: " << last_duration_generate_b
                << "s" << std::endl;
      double ops_generate_b =
          static_cast<double>(grid->getSize() * trainingData.getNrows() *
                              (6 * dimension + 1)) *
          1E-9;
      std::cout << "ops_generate_b: " << ops_generate_b << std::endl;
      double flops_generate_b = ops_generate_b / last_duration_generate_b;
      std::cout << "flops_generate_b: " << flops_generate_b << " GFLOPS"
                << std::endl;

      if (i == 0) {
        result_timings << last_duration_generate_b << "; " << flops_generate_b
                       << "; ";
      }

      if (weight_support_coarsening && i == 0) {
        // std::cout << "b: ";
        // for (size_t u = 0; u < b.size(); u += 1) {
        //   if (u > 0) {
        //     std::cout << ", ";
        //   }
        //   std::cout << b[u];
        // }

        std::cout << std::endl;
        size_t old_size = alpha.getSize();
        sgpp::base::WeightSupportCoarseningFunctor
            weight_support_coarsen_functor(b,
                                           weight_support_coarsening_threshold);
        grid_generator.coarsen(weight_support_coarsen_functor, b);
        alpha.resize(grid->getSize());
        std::cout << "b-based coarsening: removed "
                  << (old_size - grid->getSize())
                  << " grid points, remaining: " << grid->getSize()
                  << std::endl;
        operation_mult = std::unique_ptr<
            datadriven::DensityOCLMultiPlatform::OperationDensity>(
            datadriven::createDensityOCLMultiPlatformConfigured(
                *grid, dimension, lambda, configFileName));
      }

      std::cout << "Solving density SLE" << std::endl;
      solver->solve(*operation_mult, alpha, b, false, true);

      // std::cout << "Grid points after refinement step: " << grid->getSize()
      // << std::endl;

      size_t iterations = solver->getNumberIterations();
      size_t act_it = iterations + 1 + (iterations / 50);
      std::cout << "act_it: " << act_it << std::endl;

      double acc_duration_density = operation_mult->getAccDurationDensityMult();
      std::cout << "acc_duration_density: " << acc_duration_density << "s"
                << std::endl;
      double ops_density = static_cast<double>(std::pow(grid->getSize(), 2) *
                                               act_it * (14 * dimension + 2)) *
                           1E-9;
      std::cout << "ops_density: " << ops_density << " GOps" << std::endl;
      double flops_density = ops_density / acc_duration_density;
      std::cout << "flops_density: " << flops_density << " GFLOPS" << std::endl;

      if (i == 0) {
        result_timings << acc_duration_density << "; " << flops_density << "; ";
      }
    }
    density_timer_stop = std::chrono::system_clock::now();
    std::chrono::duration<double> density_elapsed_seconds =
        density_timer_stop - density_timer_start;
    std::cout << "density_duration_total: " << density_elapsed_seconds.count()
              << "s" << std::endl;

    // Output final rhs values
    if (write_rhs) {
      std::ofstream out_rhs(file_prefix + "_rhs.csv");
      for (size_t i = 0; i < grid->getSize(); i += 1) {
        out_rhs << b[i] << std::endl;
      }
      out_rhs.close();
    }
    // Output final coefficients
    if (write_density_grid) {
      std::ofstream out_grid_coord(file_prefix + "_grid_coord.csv");
      auto &storage = grid->getStorage();
      for (size_t i = 0; i < grid->getSize(); i++) {
        sgpp::base::HashGridPoint point = storage.getPoint(i);
        for (size_t d = 0; d < dimension; d++) {
          if (d > 0) {
            out_grid_coord << ", ";
          }
          out_grid_coord << storage.getCoordinate(point, d);
        }
        out_grid_coord << std::endl;
      }
      out_grid_coord.close();

      alpha.toFile(file_prefix + "_density_coef.serialized");

      std::ofstream out_grid(file_prefix + "_density_grid.serialized");
      grid->serialize(out_grid);
    }
  }

  double max = alpha.max();
  double min = alpha.min();
  std::cout << "surplus min: " << min << " max: " << max << std::endl;

  if (write_evaluated_density_full_grid) {
    std::cout << "Creating regular grid to evaluate sparse grid density "
                 "function on..."
              << std::endl;
    double h = 1.0 / std::pow(2.0, eval_grid_level); // 2^-eval_grid_level
    size_t dim_grid_points = (1 << eval_grid_level) + 1;
    size_t total_grid_points = 1;
    for (size_t d = 0; d < dimension; d++) {
      total_grid_points *= dim_grid_points;
    }
    std::cout << "total density evaluation full grid grid points: "
              << total_grid_points << std::endl;
    if (total_grid_points > 1E6) {
      std::cerr << "warning: density full grid evaluation might take very "
                   "long, this is a potential input error"
                << std::endl;
    }

    base::DataMatrix evaluationPoints(total_grid_points, dimension);

    size_t linearIndex = 0;
    std::vector<double> eval_point_enum(dimension);
    for (size_t d = 0; d < dimension; d++) {
      eval_point_enum[d] = 0;
    }
    // do first point seperately
    for (size_t d = 0; d < dimension; d++) {
      double x = static_cast<double>(eval_point_enum[d]) * h;
      evaluationPoints(linearIndex, d) = x;
    }
    linearIndex += 1;

    size_t dim_index = 0;
    while (dim_index < dimension) {
      if (eval_point_enum[dim_index] + 1 < dim_grid_points) {
        eval_point_enum[dim_index] += 1;
        for (size_t d = 0; d < dim_index; d++) {
          eval_point_enum[d] = 0;
        }
        dim_index = 0;

        for (size_t d = 0; d < dimension; d++) {
          double x = static_cast<double>(eval_point_enum[d]) * h;
          evaluationPoints(linearIndex, d) = x;
        }
        linearIndex += 1;
      } else {
        dim_index += 1;
      }
    }

    datadriven::OperationMultipleEvalConfiguration configuration(
        datadriven::OperationMultipleEvalType::STREAMING,
        datadriven::OperationMultipleEvalSubType::DEFAULT);

    std::cout << "Creating multieval operation" << std::endl;
    std::unique_ptr<base::OperationMultipleEval> eval(
        op_factory::createOperationMultipleEval(*grid, evaluationPoints,
                                                configuration));

    base::DataVector results(evaluationPoints.getNrows());
    std::cout << "Evaluating at evaluation grid points" << std::endl;
    eval->mult(alpha, results);

    std::ofstream out_density(file_prefix + std::string("_density_eval.csv"));
    out_density.precision(20);
    for (size_t eval_index = 0; eval_index < evaluationPoints.getNrows();
         eval_index += 1) {
      for (size_t d = 0; d < dimension; d += 1) {
        out_density << evaluationPoints[eval_index * dimension + d] << ", ";
      }
      out_density << results[eval_index];
      out_density << std::endl;
    }
    out_density.close();
  }

  std::vector<int64_t> graph;

  if (sampling_chunk_size == 0) {
    sampling_chunk_size = trainingData.getNrows();
  }

  sgpp::datadriven::OperationNearestNeighborSampled knn_op(true);
  if (reuse_knn_graph.compare("") != 0) {
    std::cout << "reusing knn graph" << std::endl;
    std::chrono::time_point<std::chrono::system_clock> reuse_graph_start,
        reuse_graph_stop;
    reuse_graph_start = std::chrono::system_clock::now();
    graph = knn_op.read_csv(reuse_knn_graph);
    reuse_graph_stop = std::chrono::system_clock::now();
    std::chrono::duration<double> reuse_graph_seconds =
        reuse_graph_stop - reuse_graph_start;
    std::cout << "loading graph (reuse) duration: "
              << reuse_graph_seconds.count() << "s" << std::endl;
  } else {
    std::cout << "Starting graph creation..." << std::endl;

    if (knn_algorithm.compare("lsh_cuda") == 0) {
#ifdef LSHKNN_WITH_CUDA // purely for the compiler - program exits earlier
                        // anyway if
                        // this is not the case
      graph = knn_op.knn_lsh_cuda(dimension, trainingData, k, lsh_tables,
                                  lsh_hashes, lsh_w);
      double lsh_duration = knn_op.get_last_duration();

      std::cout << "lsh_cuda duration: " << lsh_duration << "s" << std::endl;

      result_timings << lsh_duration << "; 0.0;";
#endif
    } else if (knn_algorithm.compare("lsh_ocl") == 0) {
#ifdef LSHKNN_WITH_OPENCL // purely for the compiler - program exits earlier
                          // anyway if
                          // this is not the case
      graph = knn_op.knn_lsh_opencl(dimension, trainingData, k, configFileName,
                                    lsh_tables, lsh_hashes, lsh_w);
      double lsh_duration = knn_op.get_last_duration();

      std::cout << "lsh_ocl duration: " << lsh_duration << "s" << std::endl;

      result_timings << lsh_duration << "; 0.0;";
#endif
    } else if (knn_algorithm.compare("naive_ocl") == 0) {
      graph = knn_op.knn_naive_ocl(dimension, trainingData, k, configFileName);
      double last_duration_create_graph = knn_op.get_last_duration();
      std::cout << "last_duration_create_graph: " << last_duration_create_graph
                << "s" << std::endl;

      double ops_create_graph =
          static_cast<double>(std::pow(trainingData.getNrows(), 2) * 3 *
                              dimension) *
          1E-9;
      std::cout << "ops_create_graph: " << ops_create_graph << " GOps"
                << std::endl;
      double flops_create_graph = ops_create_graph / last_duration_create_graph;
      std::cout << "flops_create_graph: " << flops_create_graph << " GFLOPS"
                << std::endl;

      result_timings << last_duration_create_graph << "; " << flops_create_graph
                     << "; ";
    } else if (knn_algorithm.compare("naive") == 0) {
#ifdef USE_LSHKNN // program exits earlier anyway if this is not the case
      graph = knn_op.knn_naive(k);
      double naive_duration = knn_op.get_last_duration();

      graph = knn_op.knn_naive(dimension, trainingData, k);
      double naive_duration = knn_op.get_last_duration();
      std::cout << "naive_duration: " << naive_duration << "s" << std::endl;
      result_timings << naive_duration << "; 0.0;";
#endif
    }

    if (write_knn_graph) {
      knn_op.write_graph_file(file_prefix + "_graph.csv", graph, k);
    }

    if (variables_map.count("compare_knn_csv_file_name") > 0) {
      std::vector<int64_t> neighbors_reference =
          knn_op.read_csv(compare_knn_csv_file_name);
      std::vector<int64_t> graph_converted;
      graph_converted = std::vector<int64_t>(graph.begin(), graph.end());
      double acc_assigned = knn_op.test_accuracy(
          neighbors_reference, graph_converted, trainingData.getNrows(), k);
      double acc_distance = knn_op.test_distance_accuracy(
          trainingData, neighbors_reference, graph_converted,
          trainingData.getNrows(), dimension, k);
      std::cout << "knn correctly assigned: " << acc_assigned << std::endl;
      std::cout << "knn distance error: " << acc_distance << std::endl;
    }
  }

  auto print_knn_graph = [&trainingData, k](std::string filename,
                                            std::vector<int64_t> &graph) {
    std::ofstream out_graph(filename);
    for (size_t i = 0; i < trainingData.getNrows(); ++i) {
      bool first = true;
      for (size_t j = 0; j < k; ++j) {
        if (graph[i * k + j] == -1) {
          continue;
        }
        if (!first) {
          out_graph << ", ";
        } else {
          first = false;
        }
        out_graph << graph[i * k + j];
      }
      out_graph << std::endl;
    }
    out_graph.close();
  };

  {
    std::cout << "Pruning graph..." << std::endl;

    std::unique_ptr<
        sgpp::datadriven::DensityOCLMultiPlatform::OperationPruneGraphOCL>
        operation_prune(sgpp::datadriven::pruneNearestNeighborGraphConfigured(
            *grid, dimension, alpha, trainingData, threshold, k,
            configFileName));
    operation_prune->prune_graph(graph);

    double last_duration_prune_graph = operation_prune->getLastDuration();
    std::cout << "last_duration_prune_graph: " << last_duration_prune_graph
              << "s" << std::endl;

    // middlepoint between node and neighbor ops
    double ops_prune_graph =
        static_cast<double>(trainingData.getNrows() * grid->getSize() *
                            (k + 1) * (6 * dimension + 2)) *
        1E-9;
    std::cout << "ops_prune_graph: " << ops_prune_graph << " GOps" << std::endl;
    double flops_prune_graph = ops_prune_graph / last_duration_prune_graph;
    std::cout << "flops_prune_graph: " << flops_prune_graph << " GFLOPS"
              << std::endl;

    result_timings << last_duration_prune_graph << "; " << flops_prune_graph
                   << "; ";

    if (write_pruned_knn_graph) {
      std::chrono::time_point<std::chrono::system_clock>
          write_pruned_knn_graph_start, write_pruned_knn_graph_stop;
      write_pruned_knn_graph_start = std::chrono::system_clock::now();
      print_knn_graph(file_prefix + "_graph_pruned.csv", graph);
      write_pruned_knn_graph_stop = std::chrono::system_clock::now();
      std::chrono::duration<double> write_pruned_knn_graph_seconds =
          write_pruned_knn_graph_stop - write_pruned_knn_graph_start;
      std::cout << "write pruned knn graph duration:"
                << write_pruned_knn_graph_seconds.count() << "s" << std::endl;
    }
  }

  {
    std::cout << "Finding clusters..." << std::endl;

    std::chrono::time_point<std::chrono::system_clock> find_cluster_timer_start;
    std::chrono::time_point<std::chrono::system_clock> find_cluster_timer_stop;
    find_cluster_timer_start = std::chrono::system_clock::now();

    std::vector<int64_t> node_cluster_map;
    std::vector<std::vector<int64_t>> clusters;
    sgpp::datadriven::clustering::find_clusters(graph, k, node_cluster_map,
                                                clusters);

    // sgpp::datadriven::clustering::connected_components(graph, k,
    // node_cluster_map, clusters,
    //                                                    connected_components_k_factor);

    find_cluster_timer_stop = std::chrono::system_clock::now();

    std::chrono::duration<double> find_cluster_elapsed_seconds =
        find_cluster_timer_stop - find_cluster_timer_start;
    std::cout << "find_cluster_duration_total: "
              << find_cluster_elapsed_seconds.count() << "s" << std::endl;

    std::cout << "detected clusters: " << clusters.size() << std::endl;
    int64_t sum_datapoints = 0;
    for (size_t i = 0; i < clusters.size(); i += 1) {
      if (print_cluster_sizes > 0 &&
          static_cast<int64_t>(clusters[i].size()) > print_cluster_sizes) {
        std::cout << "size cluster i: " << i << " -> " << clusters[i].size()
                  << std::endl;
      }
      sum_datapoints += clusters[i].size();
    }
    std::cout << "datapoints in clusters: " << sum_datapoints << std::endl;

    if (target_clusters > 0) {
      double score =
          static_cast<double>(
              target_clusters -
              std::min(target_clusters,
                       std::abs(target_clusters -
                                static_cast<int64_t>(clusters.size())))) *
          (static_cast<double>(sum_datapoints) /
           static_cast<double>(trainingData.getNrows()));
      std::cout << "score: " << score << std::endl;
    }

    // std::cout << "clusters:" << std::endl;
    // for (size_t i = 0; i < clusters.size(); i += 1) {
    //   std::cout << "c_i: " << i << " -> ";
    //   for (size_t j = 0; j < clusters[i].size(); j += 1) {
    //     int64_t other_index = clusters[i][j];
    //     if (j > 0) {
    //       std::cout << ", ";
    //     }
    //     std::cout << other_index;
    //   }
    //   std::cout << std::endl;
    // }

    if (write_cluster_map) {
      std::chrono::time_point<std::chrono::system_clock>
          write_cluster_map_start, write_cluster_map_stop;
      write_cluster_map_start = std::chrono::system_clock::now();
      std::ofstream out_cluster_map(file_prefix + "_cluster_map.csv");
      for (size_t i = 0; i < trainingData.getNrows(); ++i) {
        out_cluster_map << node_cluster_map[i] << std::endl;
      }
      out_cluster_map.close();
      write_cluster_map_stop = std::chrono::system_clock::now();
      std::chrono::duration<double> write_cluster_map_seconds =
          write_cluster_map_stop - write_cluster_map_start;
      std::cout << "write cluster map duration:"
                << write_cluster_map_seconds.count() << "s" << std::endl;
    }
  }

  {
    auto total_timer_stop = std::chrono::system_clock::now();
    std::chrono::duration<double> total_elapsed_seconds =
        total_timer_stop - total_timer_start;
    std::chrono::duration<double> total_duration_without_disk_elapsed_seconds =
        total_timer_stop - total_timer_start_without_disk;
    double total_duration = total_elapsed_seconds.count();
    double total_duration_without_disk =
        total_duration_without_disk_elapsed_seconds.count();
    std::cout << "total_duration_without_disk: " << total_duration_without_disk
              << std::endl;
    std::cout << "total_duration: " << total_duration << std::endl;
    result_timings << total_duration_without_disk << "; " << total_duration
                   << std::endl;
  }
  std::cout << std::endl << "all done!" << std::endl;
}
