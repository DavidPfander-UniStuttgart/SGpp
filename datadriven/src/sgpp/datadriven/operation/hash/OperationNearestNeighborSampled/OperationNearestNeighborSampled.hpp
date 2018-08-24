#pragma once

#include "sgpp/base/datatypes/DataMatrix.hpp"
#include "sgpp/globaldef.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>

namespace sgpp {
namespace datadriven {

class OperationNearestNeighborSampled {
private:
  base::DataMatrix dataset;
  base::DataMatrix dataset_copy;
  size_t dataset_count;
  size_t dim;

  double last_duration_s;

  bool verbose;

  base::DataMatrix sample_dataset(size_t chunk_size, size_t chunk_count);

  double l2_dist(const base::DataMatrix &dataset, const size_t first_index,
                 const size_t second_index);

  std::tuple<size_t, double>
  far_neighbor(size_t k, std::vector<int> &final_graph,
               std::vector<double> &final_graph_distances,
               size_t first_index); //, size_t &far_neighbor_index, double
                                    // &far_neighbor_distance

  void randomize(base::DataMatrix &dataset, std::vector<int> &indices_map);

  base::DataMatrix undo_randomize(base::DataMatrix &dataset,
                                  std::vector<int> &indices_map);

  void merge_knn(size_t k, base::DataMatrix &dataset, size_t chunk_first_index,
                 size_t chunk_range, std::vector<int> &final_graph,
                 std::vector<double> &final_graph_distances,
                 std::vector<int> &partial_graph,
                 std::vector<int> &indices_map);

public:
  OperationNearestNeighborSampled(base::DataMatrix dataset, size_t dim,
                                  bool verbose = false);

  std::vector<int32_t> knn_lsh(uint32_t k, uint64_t lsh_tables,
                               uint64_t lsh_hashes, double lsh_w);

  std::vector<int32_t> knn_naive(uint32_t k);

  std::vector<int32_t> knn_naive(base::DataMatrix &local_dataset, uint32_t k);

  std::vector<int32_t> knn_ocl(uint32_t k, std::string configFileName);

  std::vector<int32_t> knn_naive_sampling(uint32_t k, uint32_t input_chunk_size,
                                          uint32_t randomize_count);

  double get_last_duration();

  double test_accuracy(const std::vector<int> correct,
                       const std::vector<int> result, const int size,
                       const int k);

  double test_distance_accuracy(const std::vector<double> data,
                                const std::vector<int> correct,
                                const std::vector<int> result, const int size,
                                const int dim, const int k);

  inline std::vector<std::string> split(const std::string &s, char delim) {
    std::stringstream ss(s);
    std::string item;
    std::vector<std::string> r;

    while (std::getline(ss, item, delim)) {
      r.push_back(item);
    }

    return r;
  }

  std::vector<int> read_csv(const std::string &csv_file_name);

  void write_graph_file(const std::string &graph_file_name,
                        std::vector<int> &graph, uint64_t k);
};

} // namespace datadriven
} // namespace sgpp
