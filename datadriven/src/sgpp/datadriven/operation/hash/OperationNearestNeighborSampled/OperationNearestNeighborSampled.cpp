#include "OperationNearestNeighborSampled.hpp"
#include "KNNFactory.hpp"
#include "sgpp/datadriven/operation/hash/OperationCreateGraphOCL/OpFactory.hpp"
#include "sgpp/globaldef.hpp"
#include <algorithm>
#include <cassert>

namespace sgpp {
namespace datadriven {

size_t to_track = 0;
size_t cur_index = 0;
bool found;

OperationNearestNeighborSampled::OperationNearestNeighborSampled(bool verbose)
    : verbose(verbose) {}

std::vector<int64_t> OperationNearestNeighborSampled::knn_lsh(
    size_t dim, sgpp::base::DataMatrix &dataset, uint32_t k,
    uint64_t lsh_tables, uint64_t lsh_hashes, double lsh_w) {

  dataset.transpose();
  std::unique_ptr<lshknn::KNN> lsh(lshknn::create_knn_lsh(
      dataset, dataset.getNcols(), dim, lsh_tables, lsh_hashes, lsh_w));
  std::chrono::time_point<std::chrono::system_clock> timer_lsh_start =
      std::chrono::system_clock::now();
  std::vector<int32_t> graph = lsh->kNearestNeighbors(k);
  std::chrono::time_point<std::chrono::system_clock> timer_lsh_stop =
      std::chrono::system_clock::now();
  last_duration_s =
      std::chrono::duration<double>(timer_lsh_stop - timer_lsh_start).count();

  dataset.transpose(); // undo
  // TODO: should get rid of this
  std::vector<int64_t> graph_converted(graph.begin(), graph.end());

  // std::cout << "lsh graph.size() / k = " << (graph.size() / k) << std::endl;
  return graph_converted;
}

std::vector<int64_t> OperationNearestNeighborSampled::knn_naive(
    size_t dim, base::DataMatrix &dataset, uint32_t k) {
  dataset.transpose();
  std::unique_ptr<lshknn::KNN> lsh(
      lshknn::create_knn_naive_cpu(dataset, dataset.getNcols(), dim));
  std::chrono::time_point<std::chrono::system_clock> timer_lsh_start =
      std::chrono::system_clock::now();
  std::vector<int> graph = lsh->kNearestNeighbors(k);
  std::chrono::time_point<std::chrono::system_clock> timer_lsh_stop =
      std::chrono::system_clock::now();
  last_duration_s =
      std::chrono::duration<double>(timer_lsh_stop - timer_lsh_start).count();
  dataset.transpose(); // undo
  std::vector<int64_t> converted_graph(graph.begin(), graph.end());
  return converted_graph;
}

double OperationNearestNeighborSampled::l2_dist(const size_t dim,
                                                const base::DataMatrix &dataset,
                                                const size_t first_index,
                                                const size_t second_index) {

  // if (found && first_index == cur_index) {
  // std::cout << "first_index: " << first_index << " data: ";
  // for (size_t d = 0; d < dim; d += 1) {
  //   if (d > 0)
  //     std::cout << ", ";
  //   std::cout << dataset[first_index * dim + d];
  // }
  // // std::cout << std::endl;
  // std::cout << " second_index: " << second_index << " data: ";
  // for (size_t d = 0; d < dim; d += 1) {
  //   if (d > 0)
  //     std::cout << ", ";
  //   std::cout << dataset[second_index * dim + d];
  // }
  // std::cout << std::endl;
  // }

  double dist = 0.0;
  for (size_t d = 0; d < dim; d += 1) {
    double temp =
        dataset[first_index * dim + d] - dataset[second_index * dim + d];
    dist += temp * temp;
  }
  // if (found && first_index == cur_index) {
  // std::cout << " dist: " << sqrt(dist) << std::endl;
  // }
  return sqrt(dist);
}

std::tuple<size_t, double> OperationNearestNeighborSampled::far_neighbor(
    size_t dim, size_t k, std::vector<int64_t> &graph,
    std::vector<double> &graph_distances, size_t first_index) {

  size_t far_neighbor_index;
  double far_neighbor_distance;
  if (graph[first_index * k + 0] < 0) {
    // std::cout << "in -1 case for i: " << first_index << " j: " << 0
    //           << std::endl;
    far_neighbor_distance = 2 * dim;
    far_neighbor_index = 0;
    return {far_neighbor_index, far_neighbor_distance};
  }
  far_neighbor_index = 0;
  far_neighbor_distance = graph_distances[first_index * k + 0];

  for (size_t j = 1; j < k; j += 1) {
    if (graph[first_index * k + j] < 0) {
      // nearest neigbhor of noone in hypercube [0,1]^d
      // std::cout << "in -1 case for i: " << first_index << " j: " << j
      //           << std::endl;
      far_neighbor_distance = 2 * dim;
      far_neighbor_index = j;
      break;
    }

    double dist = graph_distances[first_index * k + j];

    if (dist > far_neighbor_distance) {
      far_neighbor_distance = dist;
      far_neighbor_index = j;
    }
  }
  // std::cout << "far_neighbor_index: " << far_neighbor_index
  //           << " far_neighbor_distance: " << far_neighbor_distance <<
  //           std::endl;
  return {far_neighbor_index, far_neighbor_distance};
}

void OperationNearestNeighborSampled::merge_knn(
    size_t dim, size_t k, base::DataMatrix &chunk, size_t chunk_range,
    std::vector<int64_t> &partial_final_graph,
    std::vector<double> &partial_final_graph_distances,
    std::vector<int64_t> &partial_graph,
    std::vector<int64_t> &partial_indices_map) {
  for (size_t i = 0; i < chunk_range; i += 1) {
    auto [max_dist_index, max_dist] = far_neighbor(
        dim, k, partial_final_graph, partial_final_graph_distances, i);
    for (size_t j = 0; j < k; j += 1) {
      double dist = l2_dist(dim, chunk, i, partial_graph[i * k + j]);
      if (dist < max_dist) {
        int64_t candidate_original_index =
            partial_indices_map[partial_graph[i * k + j]];
        bool is_duplicate = false;
        for (size_t h = 0; h < k; h += 1) {
          if (partial_final_graph[i * k + h] == candidate_original_index) {
            is_duplicate = true;
            break;
          }
        }
        if (is_duplicate) {
          continue;
        }
        partial_final_graph_distances[i * k + max_dist_index] = dist;
        partial_final_graph[i * k + max_dist_index] = candidate_original_index;
        if (j != k - 1) {
          std::tie(max_dist_index, max_dist) = far_neighbor(
              dim, k, partial_final_graph, partial_final_graph_distances, i);
        }
      }
    }
  }
}

void OperationNearestNeighborSampled::randomize(
    size_t dim, base::DataMatrix &dataset, size_t k,
    std::vector<int64_t> &graph, std::vector<double> &graph_distances,
    std::vector<int64_t> &indices_map) {
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<size_t> dist(0, dataset.getNrows() - 1);
  for (size_t i = 0; i < dataset.getNrows(); i += 1) {
    size_t swap_index = dist(mt);
    // swap (partial) dataset
    for (size_t d = 0; d < dim; d += 1) {
      double temp = dataset[swap_index * dim + d];
      dataset[swap_index * dim + d] = dataset[i * dim + d];
      dataset[i * dim + d] = temp;
    }
    // swap in (partial) graph
    for (size_t j = 0; j < k; j += 1) {
      size_t temp = graph[swap_index * k + j];
      graph[swap_index * k + j] = graph[i * k + j];
      graph[i * k + j] = temp;
    }
    // swap in (partial) graph_distances
    for (size_t j = 0; j < k; j += 1) {
      double temp = graph_distances[swap_index * k + j];
      graph_distances[swap_index * k + j] = graph_distances[i * k + j];
      graph_distances[i * k + j] = temp;
    }
    // swap in (partial) indices_map
    int64_t index_temp = indices_map[swap_index];
    indices_map[swap_index] = indices_map[i];
    indices_map[i] = index_temp;
  }
}

namespace detail {

class swap_helper {
private:
  size_t dim;
  size_t k;
  std::vector<double> data_point;
  std::vector<int64_t> neigh;
  std::vector<double> neigh_dist;
  base::DataMatrix &dataset;
  std::vector<int64_t> &final_graph;
  std::vector<double> &final_graph_distances;

public:
  swap_helper(size_t dim, base::DataMatrix &dataset, size_t k,
              std::vector<int64_t> &final_graph,
              std::vector<double> &final_graph_distances)
      : dim(dim), k(k), data_point(dim), neigh(k), neigh_dist(k),
        dataset(dataset), final_graph(final_graph),
        final_graph_distances(final_graph_distances) {}

  void save_temp(const size_t index) {
    for (size_t d = 0; d < dim; d += 1) {
      this->data_point[d] = dataset[index * dim + d];
    }
    for (size_t j = 0; j < k; j += 1) {
      this->neigh[j] = final_graph[index * k + j];
    }
    for (size_t j = 0; j < k; j += 1) {
      this->neigh_dist[j] = final_graph_distances[index * k + j];
    }
  }

  void write_temp(const size_t index) {
    for (size_t d = 0; d < dim; d += 1) {
      dataset[index * dim + d] = this->data_point[d];
    }
    for (size_t j = 0; j < k; j += 1) {
      final_graph[index * k + j] = this->neigh[j];
    }
    for (size_t j = 0; j < k; j += 1) {
      final_graph_distances[index * k + j] = this->neigh_dist[j];
    }
  }
};

} // namespace detail

void OperationNearestNeighborSampled::undo_randomize(
    size_t dim, base::DataMatrix &dataset, size_t k,
    std::vector<int64_t> &graph, std::vector<double> &graph_distances,
    std::vector<int64_t> &indices_map) {
  std::array<detail::swap_helper, 2> temps{
      detail::swap_helper{dim, dataset, k, graph, graph_distances},
      detail::swap_helper{dim, dataset, k, graph, graph_distances}};
  for (size_t i = 0; i < indices_map.size(); i += 1) {
    // test if already processd as part of a cycle
    if (indices_map[i] == -1) {
      continue;
    }
    int64_t cur_index = i;
    size_t save_temp = 0;
    size_t write_temp = 1;

    temps[save_temp].save_temp(cur_index);
    int64_t next_index = indices_map[cur_index];
    std::swap(save_temp, write_temp);
    // now can assume that cur_index was already processed
    while (indices_map[cur_index] != -1) {

      next_index = indices_map[cur_index];
      // save values of next index
      temps[save_temp].save_temp(next_index);
      // overwrite next index with values of current index (saved in last
      // round)
      temps[write_temp].write_temp(next_index);

      indices_map[cur_index] = -1;
      cur_index = next_index;
      std::swap(save_temp, write_temp);
    }
  }
}

std::tuple<base::DataMatrix, std::vector<int64_t>, std::vector<double>,
           std::vector<int64_t>>
OperationNearestNeighborSampled::extract_chunk(
    size_t dim, size_t k, size_t chunk_first_index, size_t chunk_range,
    base::DataMatrix &dataset, std::vector<int64_t> &final_graph,
    std::vector<double> &final_graph_distances,
    std::vector<int64_t> &indices_map) {
  base::DataMatrix chunk(chunk_range, dim); // TODO: padding!
  for (size_t i = 0; i < chunk_range; i += 1) {
    for (size_t d = 0; d < dim; d += 1) {
      chunk[i * dim + d] = dataset[(chunk_first_index + i) * dim + d];
    }
  }
  std::vector<int64_t> partial_final_graph(chunk_range * k);
  for (size_t i = 0; i < chunk_range; i += 1) {
    for (size_t j = 0; j < k; j += 1) {
      partial_final_graph[i * k + j] =
          final_graph[(chunk_first_index + i) * k + j];
    }
  }
  std::vector<double> partial_final_graph_distances(chunk_range * k);
  for (size_t i = 0; i < chunk_range; i += 1) {
    for (size_t j = 0; j < k; j += 1) {
      partial_final_graph_distances[i * k + j] =
          final_graph_distances[(chunk_first_index + i) * k + j];
    }
  }
  std::vector<int64_t> partial_indices_map(chunk_range);
  for (size_t i = 0; i < chunk_range; i += 1) {
    partial_indices_map[i] = indices_map[chunk_first_index + i];
  }

  return {chunk, partial_final_graph, partial_final_graph_distances,
          partial_indices_map};
}

void OperationNearestNeighborSampled::merge_chunk(
    size_t k, size_t chunk_first_index, size_t chunk_range,
    std::vector<int64_t> &final_graph,
    std::vector<double> &final_graph_distances,
    std::vector<int64_t> &partial_final_graph,
    std::vector<double> &partial_final_graph_distances) {
  for (size_t i = 0; i < chunk_range; i += 1) {
    for (size_t j = 0; j < k; j += 1) {
      final_graph[(chunk_first_index + i) * k + j] =
          partial_final_graph[i * k + j];
    }
  }
  for (size_t i = 0; i < chunk_range; i += 1) {
    for (size_t j = 0; j < k; j += 1) {
      final_graph_distances[(chunk_first_index + i) * k + j] =
          partial_final_graph_distances[i * k + j];
    }
  }
}

std::vector<int64_t> OperationNearestNeighborSampled::knn_lsh_sampling(
    size_t dim, sgpp::base::DataMatrix &dataset, uint32_t k,
    uint32_t input_chunk_size, uint32_t randomize_count, uint64_t lsh_tables,
    uint64_t lsh_hashes, double lsh_w) {

  auto chunk_knn =
      std::bind(&OperationNearestNeighborSampled::knn_lsh, this,
                std::placeholders::_1, std::placeholders::_2,
                std::placeholders::_3, lsh_tables, lsh_hashes, lsh_w);
  return knn_sampling(dim, dataset, k, input_chunk_size, randomize_count,
                      chunk_knn);
}

std::vector<int64_t> OperationNearestNeighborSampled::knn_naive_sampling(
    size_t dim, sgpp::base::DataMatrix &dataset, uint32_t k,
    uint32_t input_chunk_size, uint32_t randomize_count) {

  auto chunk_knn = std::bind(&OperationNearestNeighborSampled::knn_naive, this,
                             std::placeholders::_1, std::placeholders::_2,
                             std::placeholders::_3);
  return knn_sampling(dim, dataset, k, input_chunk_size, randomize_count,
                      chunk_knn);
}

std::vector<int64_t> OperationNearestNeighborSampled::knn_sampling(
    size_t dim, sgpp::base::DataMatrix &dataset, uint32_t k,
    uint32_t input_chunk_size, uint32_t randomize_count,
    std::function<std::vector<int64_t>(size_t, sgpp::base::DataMatrix &,
                                       uint32_t)>
        chunk_knn) {
  std::chrono::time_point<std::chrono::system_clock> timer_start =
      std::chrono::system_clock::now();
  size_t dataset_count = dataset.getNrows();

  size_t total_input_chunks = dataset_count / input_chunk_size;
  if (total_input_chunks * input_chunk_size < dataset_count) {
    total_input_chunks += 1;
  }
  // std::cout << "total_input_chunks: " << total_input_chunks << std::endl;

  std::vector<int64_t> indices_map(dataset_count);
  for (size_t i = 0; i < dataset_count; i += 1) {
    indices_map[i] = i;
  }

  std::vector<int64_t> final_graph(dataset_count * k, -1);
  std::vector<double> final_graph_distances(dataset_count * k, 3 * dim);

  randomize(dim, dataset, k, final_graph, final_graph_distances, indices_map);

  for (size_t random_iteration = 0; random_iteration < randomize_count;
       random_iteration += 1) {
    for (size_t chunk_index = 0; chunk_index < total_input_chunks;
         chunk_index += 1) {
      // extract chunk from dataset
      size_t chunk_first_index = chunk_index * input_chunk_size;
      size_t chunk_last_index =
          std::min((chunk_index + 1) * input_chunk_size, dataset.size() / dim);
      size_t chunk_range = chunk_last_index - chunk_first_index;
      if (verbose) {
        std::cout << "chunk from: " << chunk_first_index
                  << " to: " << chunk_last_index << " range: " << chunk_range
                  << std::endl;
      }
      auto [chunk, partial_final_graph, partial_final_graph_distances,
            partial_indices_map] =
          extract_chunk(dim, k, chunk_first_index, chunk_range, dataset,
                        final_graph, final_graph_distances, indices_map);

      // sgpp::base::DataMatrix chunk_copy(chunk);

      // std::cout << "chunk before:" << std::endl;
      // for (size_t i = 0; i < chunk_range; i += 1) {
      //   for (size_t j = 0; j < dim; j += 1) {
      //     if (j > 0)
      //       std::cout << ", ";
      //     std::cout << chunk[i * k + j];
      //   }
      //   std::cout << std::endl;
      // }

      // std::vector<int64_t> partial_graph = knn_naive(dim, chunk, k);
      std::vector<int64_t> partial_graph = chunk_knn(dim, chunk, k);

      // std::cout << "chunk after:" << std::endl;
      // for (size_t i = 0; i < chunk_range; i += 1) {
      //   for (size_t j = 0; j < dim; j += 1) {
      //     if (j > 0)
      //       std::cout << ", ";
      //     std::cout << chunk[i * k + j];
      //   }
      //   std::cout << std::endl;
      // }

      // for (size_t i = 0; i < chunk_range; i += 1) {
      //   for (size_t j = 0; j < dim; j += 1) {
      //     assert(chunk[i * dim + j] == chunk_copy[i * dim + j]);
      //   }
      // }

      merge_knn(dim, k, chunk, chunk_range, partial_final_graph,
                partial_final_graph_distances, partial_graph,
                partial_indices_map);

      merge_chunk(k, chunk_first_index, chunk_range, final_graph,
                  final_graph_distances, partial_final_graph,
                  partial_final_graph_distances);
      if (verbose) {
        double acc_dist = 0.0;
        for (size_t i = 0; i < chunk_range; i += 1) {
          for (size_t j = 0; j < k; j += 1) {
            acc_dist += partial_final_graph_distances[i * k + j];
          }
        }
        std::cout << "chunk acc_dist: " << acc_dist << std::endl;
      }
    }
    randomize(dim, dataset, k, final_graph, final_graph_distances, indices_map);
  }

  undo_randomize(dim, dataset, k, final_graph, final_graph_distances,
                 indices_map);

  std::chrono::time_point<std::chrono::system_clock> timer_stop =
      std::chrono::system_clock::now();
  last_duration_s =
      std::chrono::duration<double>(timer_stop - timer_start).count();
  return final_graph;
}

std::vector<int64_t> OperationNearestNeighborSampled::knn_ocl(
    size_t dim, sgpp::base::DataMatrix &dataset, uint32_t k,
    std::string configFileName) {
  size_t dataset_count = dataset.getNrows();
  // TODO: need to convert operation graph to larger than int
  std::vector<int32_t> graph(dataset_count * k, -1);
  auto operation_graph = std::unique_ptr<
      sgpp::datadriven::DensityOCLMultiPlatform::OperationCreateGraphOCL>(
      sgpp::datadriven::createNearestNeighborGraphConfigured(dataset, k, dim,
                                                             configFileName));

  operation_graph->create_graph(graph);

  last_duration_s = operation_graph->getLastDuration();
  std::vector<int64_t> converted_graph(graph.begin(), graph.end());
  return converted_graph;
}

double OperationNearestNeighborSampled::get_last_duration() {
  return last_duration_s;
}

double OperationNearestNeighborSampled::test_accuracy(
    const std::vector<int64_t> &correct, const std::vector<int64_t> &result,
    const int dataset_count, const int k) {
  int count = 0;
  int total = 0;
  for (int s = 0; s < dataset_count; s += 1) {
    for (int c = 0; c < k; c += 1) {
      for (int r = 0; r < k; r += 1) {
        if (correct[s * k + c] == result[s * k + r]) {
          count += 1;
          break;
        }
      }
      total += 1;
    }
  }
  return static_cast<double>(count) / static_cast<double>(total);
}

double OperationNearestNeighborSampled::test_distance_accuracy(
    const std::vector<double> &data, const std::vector<int64_t> &correct,
    const std::vector<int64_t> &result, const int dataset_count, const int dim,
    const int k) {
  double dist_sum_correct = 0.0;
  double dist_sum_lsh = 0.0;
  for (int i = 0; i < dataset_count; i += 1) {
    for (int j = 0; j < k; j += 1) {
      double dist_correct = 0.0;
      for (int d = 0; d < dim; d += 1) {
        dist_correct +=
            (data[i * dim + d] - data[correct[k * i + j] * dim + d]) *
            (data[i * dim + d] - data[correct[k * i + j] * dim + d]);
      }
      dist_sum_correct += sqrt(dist_correct);
    }
    for (int j = 0; j < k; j += 1) {
      double dist_lsh = 0.0;
      for (int d = 0; d < dim; d += 1) {
        dist_lsh += (data[i * dim + d] - data[result[k * i + j] * dim + d]) *
                    (data[i * dim + d] - data[result[k * i + j] * dim + d]);
      }
      dist_sum_lsh += sqrt(dist_lsh);
    }
  }
  // std::cout << "dist_sum_correct: " << dist_sum_correct << std::endl;
  // std::cout << "dist_sum_lsh: " << dist_sum_lsh << std::endl;
  return std::abs(dist_sum_lsh - dist_sum_correct) / dist_sum_correct;
}

std::vector<int64_t>
OperationNearestNeighborSampled::read_csv(const std::string &csv_file_name) {
  std::vector<int64_t> neighbors_csv;
  std::ifstream csv_file(csv_file_name, std::ifstream::in);
  std::string line;
  while (csv_file.good()) {
    std::getline(csv_file, line);
    std::vector<std::string> splitted = split(line, ',');
    for (size_t i = 0; i < splitted.size(); i++) {
      std::stringstream ss;
      ss << splitted[i];
      int value;
      ss >> value;
      neighbors_csv.push_back(value);
    }
  }
  return neighbors_csv;
}

void OperationNearestNeighborSampled::write_graph_file(
    const std::string &graph_file_name, std::vector<int64_t> &graph,
    uint64_t k) {
  std::ofstream out_graph(graph_file_name);
  for (size_t i = 0; i < graph.size() / k; i += 1) {
    bool first = true;
    for (size_t j = 0; j < k; j += 1) {
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
}

} // namespace datadriven
} // namespace sgpp
