#pragma once

#include "sgpp/base/datatypes/DataMatrix.hpp"
#include "sgpp/globaldef.hpp"

#include <cmath>
#include <fstream>
#include <functional>
#include <sstream>
#include <vector>

namespace sgpp {
namespace datadriven {

class OperationNearestNeighborSampled {
private:
  // base::DataMatrix &dataset;
  // size_t dataset_count;
  // size_t dim;

  double last_duration_s;

  bool verbose;

  // base::DataMatrix sample_dataset(size_t chunk_size, size_t chunk_count);

  double l2_dist(size_t dim, const base::DataMatrix &dataset,
                 const size_t first_index, const size_t second_index);

  std::tuple<size_t, double>
  far_neighbor(size_t dim, size_t k, std::vector<int64_t> &final_graph,
               std::vector<double> &final_graph_distances, size_t first_index);

public:
  OperationNearestNeighborSampled(bool verbose = false);

#ifdef LSHKNN_WITH_CUDA
  std::vector<int64_t> knn_lsh_cuda(size_t dim, sgpp::base::DataMatrix &dataset,
                                    uint32_t k, uint64_t lsh_tables,
                                    uint64_t lsh_hashes, double lsh_w);
#endif

  std::vector<int64_t>
  knn_lsh_opencl(size_t dim, sgpp::base::DataMatrix &dataset, uint32_t k,
                 const std::string &opencl_configuration_file,
                 uint64_t lsh_tables, uint64_t lsh_hashes, double lsh_w);

  std::vector<int64_t> knn_naive(size_t dim, sgpp::base::DataMatrix &dataset,
                                 uint32_t k);

  std::vector<int64_t> knn_naive_ocl(size_t dim,
                                     sgpp::base::DataMatrix &dataset,
                                     uint32_t k, std::string configFileName);

#ifdef LSHKNN_WITH_CUDA
  std::vector<int64_t>
  knn_lsh_cuda_sampling(size_t dim, sgpp::base::DataMatrix &dataset, uint32_t k,
                        uint32_t input_chunk_size, uint32_t randomize_count,
                        uint64_t lsh_tables, uint64_t lsh_hashes, double lsh_w,
                        size_t rand_chunk_size = 0);
#endif

  std::vector<int64_t> knn_lsh_opencl_sampling(
      size_t dim, sgpp::base::DataMatrix &dataset, uint32_t k,
      const std::string &opencl_configuration_file, uint32_t input_chunk_size,
      uint32_t randomize_count, uint64_t lsh_tables, uint64_t lsh_hashes,
      double lsh_w, size_t rand_chunk_size = 0);

  std::vector<int64_t> knn_naive_sampling(size_t dim,
                                          sgpp::base::DataMatrix &dataset,
                                          uint32_t k, uint32_t input_chunk_size,
                                          uint32_t randomize_count,
                                          size_t rand_chunk_size = 0);

  // set rand_chunk_size to 0 to disabled randomization on the chunk level
  std::vector<int64_t> knn_sampling(
      size_t dim, sgpp::base::DataMatrix &dataset, uint32_t k,
      uint32_t input_chunk_size, uint32_t randomize_count,
      std::function<std::vector<int64_t>(size_t, sgpp::base::DataMatrix &,
                                         uint32_t)>
          chunk_knn,
      size_t rand_chunk_size = 0);

  double get_last_duration();

  void randomize(size_t dim, base::DataMatrix &dataset, size_t k,
                 std::vector<int64_t> &graph,
                 std::vector<double> &graph_distances,
                 std::vector<int64_t> &indices_map);

  void undo_randomize(size_t dim, base::DataMatrix &dataset, size_t k,
                      std::vector<int64_t> &graph,
                      std::vector<double> &graph_distances,
                      std::vector<int64_t> &indices_map);

  void randomize_chunk(size_t dim, base::DataMatrix &dataset, size_t k,
                       std::vector<int64_t> &graph,
                       std::vector<double> &graph_distances,
                       std::vector<int64_t> &indices_map,
                       size_t rand_chunk_size);

  void merge_knn(size_t dim, size_t k, base::DataMatrix &chunk,
                 size_t chunk_range, std::vector<int64_t> &partial_final_graph,
                 std::vector<double> &partial_final_graph_distances,
                 std::vector<int64_t> &partial_graph,
                 std::vector<int64_t> &partial_indices_map);

  std::tuple<base::DataMatrix, std::vector<int64_t>, std::vector<double>,
             std::vector<int64_t>>
  extract_chunk(size_t dim, size_t k, size_t chunk_first_index,
                size_t chunk_range, base::DataMatrix &dataset,
                std::vector<int64_t> &final_graph,
                std::vector<double> &final_graph_distances,
                std::vector<int64_t> &indices_map);

  void write_back_chunk(size_t k, size_t chunk_first_index, size_t chunk_range,
                        std::vector<int64_t> &final_graph,
                        std::vector<double> &final_graph_distances,
                        std::vector<int64_t> &partial_final_graph,
                        std::vector<double> &partial_final_graph_distances);

  void write_back_chunk(size_t dim, size_t k, size_t chunk_first_index,
                        size_t chunk_range, base::DataMatrix &dataset,
                        base::DataMatrix &chunk,
                        std::vector<int64_t> &final_graph,
                        std::vector<double> &final_graph_distances,
                        std::vector<int64_t> &partial_final_graph,
                        std::vector<double> &partial_final_graph_distances,
                        std::vector<int64_t> &indices_map,
                        std::vector<int64_t> &partial_indices_map);

  double test_accuracy(const std::vector<int64_t> &correct,
                       const std::vector<int64_t> &result,
                       const int dataset_count, const int k);

  double test_distance_accuracy(const std::vector<double> &data,
                                const std::vector<int64_t> &correct,
                                const std::vector<int64_t> &result,
                                const int dataset_count, const int dim,
                                const int k);

  inline std::vector<std::string> split(const std::string &s, char delim) {
    std::stringstream ss(s);
    std::string item;
    std::vector<std::string> r;

    while (std::getline(ss, item, delim)) {
      r.push_back(item);
    }

    return r;
  }

  std::vector<int64_t> read_csv(const std::string &csv_file_name);

  void write_graph_file(const std::string &graph_file_name,
                        std::vector<int64_t> &graph, uint64_t k);
};

} // namespace datadriven
} // namespace sgpp
