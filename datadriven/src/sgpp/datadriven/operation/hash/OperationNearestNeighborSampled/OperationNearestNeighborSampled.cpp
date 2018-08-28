#include "OperationNearestNeighborSampled.hpp"
#include "KNNFactory.hpp"
#include "sgpp/datadriven/operation/hash/OperationCreateGraphOCL/OpFactory.hpp"
#include "sgpp/globaldef.hpp"
#include <algorithm>
#include <cassert>

namespace sgpp {
namespace datadriven {

OperationNearestNeighborSampled::OperationNearestNeighborSampled(
    base::DataMatrix dataset, size_t dim, bool verbose)
    : dataset(dataset), dataset_copy(dataset),
      dataset_count(dataset.getNrows()), dim(dim), verbose(verbose) {
  // if (chunk_size * chunk_count < dataset.getNrows()) {
  //   chunk_count += 1;
  // }
  if (verbose) {
    std::cout << "dataset_count:" << dataset_count << std::endl;
    // std::cout << "chunk_size:" << chunk_size << std::endl;
    // std::cout << "chunk_count:" << chunk_count << std::endl;
  }
}

base::DataMatrix
OperationNearestNeighborSampled::sample_dataset(size_t chunk_size,
                                                size_t chunk_count) {
  base::DataMatrix sampled_dataset;
  sampled_dataset.resize(chunk_count * chunk_size, dim);
  std::random_device rd;
  std::mt19937 mt(rd());
  // exclude upper bound
  size_t sample_count = dataset_count / chunk_size - 1;
  std::cout << "sample_count: " << sample_count << std::endl;
  std::uniform_int_distribution<uint64_t> dist(0, sample_count);

  for (size_t chunk_index = 0; chunk_index < chunk_count; chunk_index += 1) {
    uint64_t dataset_chunk_index = dist(mt);
    std::cout << "dataset_chunk_index: " << dataset_chunk_index << std::endl;
    for (size_t i = 0; i < chunk_size; i += 1) {
      if ((dataset_chunk_index * chunk_size + i) * dim >= dataset_count * dim) {
        // use out-of-domain grid point for padding
        std::cout << "adding padding for: "
                  << ((dataset_chunk_index * chunk_size + i) * dim)
                  << std::endl;
        for (size_t d = 0; d < dim; d += 1) {
          sampled_dataset[(chunk_index * chunk_size + i) * dim + d] = dim * 1.0;
        }
      } else {
        for (size_t d = 0; d < dim; d += 1) {
          sampled_dataset[(chunk_index * chunk_size + i) * dim + d] =
              dataset[(dataset_chunk_index * chunk_size + i) * dim + d];
        }
      }
    }
  }

  // bool all_ok = true;
  // for (size_t i = 0; i < dataset_count; i++) {
  //   for (size_t d = 0; d < dim; d++) {
  //     if (sampled_dataset[i * dim + d] != dataset[i * dim + d]) {
  //       std::cout << "error for i = " << i << ", d = " << d << std::endl;
  //       all_ok = false;
  //       break;
  //     }
  //   }
  // }
  // if (all_ok) {
  //   std::cout << "sampling ok!" << std::endl;
  // } else {
  //   std::cerr << "sampling ERROR!" << std::endl;
  // }

  return sampled_dataset;
}

std::vector<int32_t>
OperationNearestNeighborSampled::knn_lsh(uint32_t k, uint64_t lsh_tables,
                                         uint64_t lsh_hashes, double lsh_w) {
  // sample
  // base::DataMatrix sampled_dataset = sample_dataset(1, dataset_count);
  base::DataMatrix sampled_dataset(dataset);

  // transpose
  sampled_dataset.transpose();
  // std::unique_ptr<lshknn::KNN> lsh(lshknn::create_knn_naive_cpu(
  //     sampled_dataset, sampled_dataset.getNcols(), dim));
  std::unique_ptr<lshknn::KNN> lsh(
      lshknn::create_knn_lsh(sampled_dataset, sampled_dataset.getNcols(), dim,
                             lsh_tables, lsh_hashes, lsh_w));
  std::chrono::time_point<std::chrono::system_clock> timer_lsh_start =
      std::chrono::system_clock::now();
  std::vector<int> graph = lsh->kNearestNeighbors(k);
  std::chrono::time_point<std::chrono::system_clock> timer_lsh_stop =
      std::chrono::system_clock::now();
  last_duration_s =
      std::chrono::duration<double>(timer_lsh_stop - timer_lsh_start).count();

  std::cout << "lsh graph.size() / k = " << (graph.size() / k) << std::endl;
  return graph;
}

std::vector<int64_t> OperationNearestNeighborSampled::knn_naive(uint32_t k) {
  return knn_naive(dataset, k);
}

std::vector<int64_t>
OperationNearestNeighborSampled::knn_naive(base::DataMatrix &local_dataset,
                                           uint32_t k) {
  base::DataMatrix transposed_dataset(local_dataset);
  transposed_dataset.transpose();
  std::unique_ptr<lshknn::KNN> lsh(lshknn::create_knn_naive_cpu(
      transposed_dataset, transposed_dataset.getNcols(), dim));
  std::chrono::time_point<std::chrono::system_clock> timer_lsh_start =
      std::chrono::system_clock::now();
  std::vector<int> graph = lsh->kNearestNeighbors(k);
  // std::cout << "graph in knn_naive:" << std::endl;
  // for (size_t i = 0; i < graph.size() / k; i += 1) {
  //   for (size_t j = 0; j < k; j += 1) {
  //     if (j > 0) {
  //       std::cout << ", ";
  //     }
  //     std::cout << graph[i * k + j];
  //   }
  //   std::cout << std::endl;
  // }
  std::chrono::time_point<std::chrono::system_clock> timer_lsh_stop =
      std::chrono::system_clock::now();
  last_duration_s =
      std::chrono::duration<double>(timer_lsh_stop - timer_lsh_start).count();
  std::vector<int64_t> converted_graph(graph.begin(), graph.end());
  // std::cout << "converted_graph:" << std::endl;
  // for (size_t i = 0; i < converted_graph.size() / k; i += 1) {
  //   for (size_t j = 0; j < k; j += 1) {
  //     if (j > 0) {
  //       std::cout << ", ";
  //     }
  //     std::cout << converted_graph[i * k + j];
  //   }
  //   std::cout << std::endl;
  // }
  return converted_graph;
}

double OperationNearestNeighborSampled::l2_dist(const base::DataMatrix &dataset,
                                                const size_t first_index,
                                                const size_t second_index) {
  double dist = 0.0;
  for (size_t d = 0; d < dim; d += 1) {
    double temp =
        dataset[first_index * dim + d] - dataset[second_index * dim + d];
    dist += temp * temp;
  }
  return sqrt(dist);
}

std::tuple<size_t, double> OperationNearestNeighborSampled::far_neighbor(
    size_t k, std::vector<int64_t> &graph, std::vector<double> &graph_distances,
    size_t first_index) {

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
    size_t k, base::DataMatrix &chunk, size_t chunk_first_index,
    size_t chunk_range, std::vector<int64_t> &partial_final_graph,
    std::vector<double> &partial_final_graph_distances,
    std::vector<int64_t> &partial_graph
    // , std::vector<int64_t> &partial_indices_map
) {
  for (size_t i = 0; i < chunk_range; i += 1) {
    // next index in graph to process (updating its neighbors)
    // first look up which index in the final graph is to be updated
    // dataset index in the randomized data
    // const int64_t index = chunk_first_index + i;
    // original index, needed to properly access the final graph
    // const int64_t original_index = indices_map[mapped_index];
    // std::cout << "final_graph index: " << original_index << " ns: ";
    // for (size_t j = 0; j < k; j += 1) {
    //   if (j > 0)
    //     std::cout << ", ";
    //   std::cout << final_graph[original_index * k + j];
    // }
    // std::cout << " ";

    // std::cout << "update cand. index: " << i << " ns: ";
    // for (size_t j = 0; j < k; j += 1) {
    //   if (j > 0)
    //     std::cout << ", ";
    //   size_t mapped_candidate_index =
    //       chunk_first_index + partial_graph[i * k + j];
    //   int unmapped_candidate_index = indices_map[mapped_candidate_index];
    //   std::cout << unmapped_candidate_index;
    // }
    // std::cout << std::endl;

    // std::cout << "mapped_chunk_index: " << mapped_chunk_index << std::endl;
    // std::cout << "mapped_dataset_index: " << mapped_dataset_index <<
    // std::endl; std::cout << "chunk index i: " << i << " -- mapped to --> "
    //           << mapped_chunk_index << std::endl;

    // size_t max_dist_index;
    // double max_dist;

    // TODO: change final_graph_distances to contain -1.0 as marker for
    // not-yet-set to get rid of final_graph
    auto [max_dist_index, max_dist] =
        far_neighbor(k, partial_final_graph, partial_final_graph_distances, i);
    // std::cout << "final_graph_distances: ";
    // for (size_t j = 0; j < k; j += 1) {
    //   if (j > 0)
    //     std::cout << ", ";
    //   std::cout << final_graph_distances[original_index * k + j];
    // }
    // std::cout << std::endl;
    // std::cout << "max_dist_index: " << max_dist_index
    //           << " max_dist: " << max_dist << std::endl;

    // ////////////////////////////////////////////////////////////
    // // calculate pre-update dist to all neighbors
    // double dist_sum = 0.0;
    // for (size_t j = 0; j < k; j += 1) {
    //   // size_t mapped_candidate_index =
    //   //     chunk_first_index + partial_graph[i * k + j];
    //   // dist_sum += l2_dist(dataset, mapped_index, mapped_candidate_index);
    //   dist_sum += final_graph_distances[original_index * k + j];
    // }
    // std::cout << "pre-update dist_sum: " << dist_sum << std::endl;
    // ////////////////////////////////////////////////////////////
    // dist_sum = 0.0;
    // for (size_t j = 0; j < k; j += 1) {
    //   if (final_graph[original_index * k + j] == -1) {
    //     dist_sum += 4.0;
    //   } else {
    //     dist_sum += l2_dist(dataset_copy, original_index,
    //                         final_graph[original_index * k + j]);
    //   }
    // }
    // ////////////////////////////////////////////////////////////
    // std::cout << "pre-update dist_sum recalc: " << dist_sum << std::endl;
    // dist_sum = 0.0;
    // for (size_t j = 0; j < k; j += 1) {
    //   if (partial_graph[i * k + j] == -1) {
    //     dist_sum += 4.0;
    //   } else {
    //     dist_sum += l2_dist(dataset, mapped_index,
    //                         chunk_first_index + partial_graph[i * k + j]);
    //   }
    // }
    // ////////////////////////////////////////////////////////////
    // std::cout << "pre-update partial graph dist_sum recalc: " << dist_sum
    //           << std::endl;

    for (size_t j = 0; j < k; j += 1) {

      // size_t mapped_candidate_index =
      //     chunk_first_index + partial_graph[i * k + j];
      // int64_t unmapped_candidate_index = indices_map[mapped_candidate_index];
      // test that the data point used actually matches
      // for (size_t dd = 0; dd < dim; dd += 1) {
      // assert(dataset[mapped_candidate_index * dim + dd] ==
      //        dataset_copy[unmapped_candidate_index * dim + dd]);
      // std::cout << "map: " << mapped_candidate_index
      //           << " unmapped: " << unmapped_candidate_index
      //           << " d map: " << dataset[mapped_candidate_index * dim + dd]
      //           << " d unmapped: "
      //           << dataset_copy[unmapped_candidate_index * dim + dd]
      //           << std::endl;
      // }
      double dist = l2_dist(chunk, i, partial_graph[i * k + j]);
      // std::cout << "dist: " << dist << " max_dist: " << max_dist <<
      // std::endl;
      if (dist < max_dist) {
        // std::cout << "update!" << std::endl;

        // std::cout << "partial_graph[" << mapped_chunk_index << " * " << k
        //           << " + " << j
        //           << "] = " << partial_graph[mapped_chunk_index * k + j]
        //           << " -- unmapped to --> " << unmapped_candidate_index
        //           << std::endl;
        int64_t candidate_index = chunk_first_index + partial_graph[i * k + j];
        bool is_duplicate = false;
        for (size_t h = 0; h < k; h += 1) {
          if (partial_final_graph[i * k + h] == candidate_index) {
            is_duplicate = true;
            break;
          }
        }
        if (is_duplicate) {
          // std::cout << "is duplicate!" << std::endl;
          continue;
        }

        // assert(partial_final_graph[i * k + max_dist_index] !=
        // candidate_index); std::cout << "non-duplicate update!" << std::endl;
        partial_final_graph_distances[i * k + max_dist_index] = dist;
        partial_final_graph[i * k + max_dist_index] = candidate_index;
        if (j != k - 1) {
          std::tie(max_dist_index, max_dist) = far_neighbor(
              k, partial_final_graph, partial_final_graph_distances, i);
        }
      }
    }
    // //////////////////////////////////////////
    // dist_sum = 0.0;
    // for (size_t j = 0; j < k; j += 1) {
    //   dist_sum += final_graph_distances[original_index * k + j];
    // }
    // std::cout << "post-update dist_sum: " << dist_sum << std::endl;
    // //////////////////////////////////////////
    // std::cout << "data point at original_index: ";
    // for (size_t jj = 0; jj < dim; jj += 1) {
    //   if (jj > 0)
    //     std::cout << ", ";
    //   std::cout << dataset_copy[original_index * dim + jj];
    // }
    // std::cout << " ";
    // for (size_t j = 0; j < k; j += 1) {
    //   std::cout << " neigh " << j << ": ";
    //   for (size_t jj = 0; jj < dim; jj += 1) {
    //     if (jj > 0)
    //       std::cout << ", ";
    //     std::cout
    //         << dataset_copy[final_graph[original_index * k + j] * dim + jj];
    //   }
    // }
    // std::cout << std::endl;
    // //////////////////////////////////////////
    // dist_sum = 0.0;
    // for (size_t j = 0; j < k; j += 1) {
    //   if (final_graph[original_index * k + j] == -1) {
    //     dist_sum += 4.0;
    //   } else {
    //     dist_sum += l2_dist(dataset_copy, original_index,
    //                         final_graph[original_index * k + j]);
    //   }
    // }
    // std::cout << "post-update dist_sum recalc: " << dist_sum << std::endl;
    // //////////////////////////////////////////
    // std::cout << "data point at mapped_index  : ";
    // for (size_t jj = 0; jj < dim; jj += 1) {
    //   if (jj > 0)
    //     std::cout << ", ";
    //   std::cout << dataset[mapped_index * dim + jj];
    // }
    // std::cout << " ";
    // for (size_t j = 0; j < k; j += 1) {
    //   std::cout << " neigh " << j << ": ";
    //   size_t neighbor_mapped_index = 0;
    //   for (size_t jj = 0; jj < dataset_count; jj += 1) {
    //     if (indices_map[jj] == final_graph[original_index * k + j]) {
    //       neighbor_mapped_index = jj;
    //       break;
    //     }
    //   }
    //   for (size_t jj = 0; jj < dim; jj += 1) {
    //     if (jj > 0)
    //       std::cout << ", ";
    //     std::cout << dataset[neighbor_mapped_index * dim + jj];
    //   }
    // }
    // std::cout << std::endl;
    // //////////////////////////////////////////
    // dist_sum = 0.0;
    // for (size_t j = 0; j < k; j += 1) {
    //   if (final_graph[original_index * k + j] == -1) {
    //     dist_sum += 4.0;
    //   } else {
    //     size_t neighbor_mapped_index = 0;
    //     for (size_t jj = 0; jj < dataset_count; jj += 1) {
    //       if (indices_map[jj] == final_graph[original_index * k + j]) {
    //         neighbor_mapped_index = jj;
    //         break;
    //       }
    //     }
    //     dist_sum += l2_dist(dataset, mapped_index, neighbor_mapped_index);
    //   }
    // }
    // std::cout << "post-update dist_sum recalc in rand. dataset: " << dist_sum
    //           << std::endl;
  }
} // namespace datadriven

void OperationNearestNeighborSampled::randomize(
    size_t k, base::DataMatrix &dataset, std::vector<int64_t> &graph,
    std::vector<double> &graph_distances, std::vector<int64_t> &indices_map) {
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
  swap_helper(size_t dim, size_t k, base::DataMatrix &dataset,
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
    size_t k, base::DataMatrix &dataset, std::vector<int64_t> &graph,
    std::vector<double> &graph_distances, std::vector<int64_t> &indices_map) {
  std::array<detail::swap_helper, 2> temps{
      detail::swap_helper{dim, k, dataset, graph, graph_distances},
      detail::swap_helper{dim, k, dataset, graph, graph_distances}};
  for (size_t i = 0; i < indices_map.size(); i += 1) {
    // test if already processd as part of a cycle
    if (indices_map[i] == -1) {
      continue;
    }
    // int64_t last_index = -1;
    int64_t cur_index = i;
    size_t save_temp = 0;
    size_t write_temp = 1;

    temps[save_temp].save_temp(cur_index);
    int64_t next_index = indices_map[cur_index];
    // std::cout << "start cur_index: " << cur_index
    //           << " next_index: " << next_index << std::endl;
    // indices_map[cur_index] = -1;
    // cur_index = next_index;
    std::swap(save_temp, write_temp);
    // now can assume that cur_index was already processed
    while (indices_map[cur_index] != -1) {

      next_index = indices_map[cur_index];
      // std::cout << "loop cur_index: " << cur_index
      //           << " next_index: " << next_index << std::endl;
      // save values of next index
      temps[save_temp].save_temp(next_index);
      // overwrite next index with values of current index (saved in last
      // round)
      temps[write_temp].write_temp(next_index);

      indices_map[cur_index] = -1;
      cur_index = next_index;
      std::swap(save_temp, write_temp);
    }
    // std::cout << "loop end" << std::endl;
  }
}

// namespace detail {
std::tuple<base::DataMatrix, std::vector<int64_t>, std::vector<double>>
OperationNearestNeighborSampled::extract_chunk(
    size_t dim, size_t k, size_t chunk_first_index, size_t chunk_range,
    base::DataMatrix &dataset, std::vector<int64_t> &final_graph,
    std::vector<double> &final_graph_distances
    // , std::vector<int64_t> &indices_map
) {
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
  // std::vector<int64_t> partial_indices_map(chunk_range);
  // for (size_t i = 0; i < chunk_range; i += 1) {
  //   partial_indices_map[i] = indices_map[chunk_first_index + i];
  // }
  return {
      chunk, partial_final_graph, partial_final_graph_distances
      // ,         partial_indices_map
  };
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

std::vector<int64_t> OperationNearestNeighborSampled::knn_naive_sampling(
    uint32_t k, uint32_t input_chunk_size, uint32_t randomize_count) {
  std::chrono::time_point<std::chrono::system_clock> timer_start =
      std::chrono::system_clock::now();

  size_t total_input_chunks = dataset_count / input_chunk_size;
  if (total_input_chunks * input_chunk_size < dataset_count) {
    total_input_chunks += 1;
  }
  std::cout << "total_input_chunks: " << total_input_chunks << std::endl;

  std::vector<int64_t> indices_map(dataset_count);
  for (size_t i = 0; i < dataset_count; i += 1) {
    indices_map[i] = i;
  }

  std::vector<int64_t> final_graph(dataset_count * k, -1);
  std::vector<double> final_graph_distances(dataset_count * k);

  randomize(k, dataset, final_graph, final_graph_distances, indices_map);

  for (size_t random_iteration = 0; random_iteration < randomize_count;
       random_iteration += 1) {
    for (size_t chunk_index = 0; chunk_index < total_input_chunks;
         chunk_index += 1) {
      // extract chunk from dataset
      size_t chunk_first_index = chunk_index * input_chunk_size;
      size_t chunk_last_index =
          std::min((chunk_index + 1) * input_chunk_size, dataset.size() / dim);
      size_t chunk_range = chunk_last_index - chunk_first_index;
      std::cout << "chunk from: " << chunk_first_index
                << " to: " << chunk_last_index << " range: " << chunk_range
                << std::endl;
      auto [chunk, partial_final_graph, partial_final_graph_distances] =
          extract_chunk(dim, k, chunk_first_index, chunk_range, dataset,
                        final_graph, final_graph_distances);

      // std::cout << "----- chunk -----" << std::endl;
      // for (size_t i = 0; i < chunk_range; i += 1) {
      //   for (size_t d = 0; d < dim; d += 1) {
      //     if (d > 0) {
      //       std::cout << ", ";
      //     }
      //     std::cout << chunk[i * dim + d];
      //   }
      //   std::cout << std::endl;
      // }
      // std::cout << "----- end chunk -----" << std::endl;
      // std::cout << "chunk.size() = " << chunk.size() << std::endl;
      std::vector<int64_t> partial_graph = knn_naive(chunk, k);

      // std::cout << "real indices:" << std::endl;
      // for (size_t i = 0; i < chunk_range; i += 1) {
      //   if (i > 0)
      //     std::cout << ", ";
      //   std::cout << indices_map[i];
      // }
      // std::cout << std::endl;

      // std::cout << "partial graph:" << std::endl;
      // for (size_t i = 0; i < chunk_range; i += 1) {
      //   std::sort(partial_graph.begin() + i * k,
      //             partial_graph.begin() + (i + 1) * k);
      //   for (size_t j = 0; j < k; j += 1) {
      //     if (j > 0)
      //       std::cout << ", ";
      //     std::cout
      //         << indices_map[chunk_first_index + partial_graph[i * k + j]];
      //   }
      //   std::cout << std::endl;
      // }
      // std::cout << "----- end partial -----" << std::endl;

      // TODO: should get rid of everything that requires the full dataset, that
      // is
      // - dataset -> node-local partial dataset (== chunk)
      // - final_graph -> node-local partial final_graph
      // - final_graph_distances -> same as final_graph
      // - indices_map -> node-local partial indices_map
      merge_knn(k, dataset, chunk_first_index, chunk_range, partial_final_graph,
                partial_final_graph_distances, partial_graph
                // , partial_indices_map

      );

      merge_chunk(k, chunk_first_index, chunk_range, final_graph,
                  final_graph_distances, partial_final_graph,
                  partial_final_graph_distances);
    }
    randomize(k, dataset, final_graph, final_graph_distances, indices_map);

    double dist_sum_lsh = 0.0;
    for (size_t i = 0; i < dataset_count; i += 1) {
      for (size_t j = 0; j < k; j += 1) {
        double dist_lsh = 0.0;
        for (size_t d = 0; d < dim; ++d) {
          dist_lsh += (dataset_copy[i * dim + d] -
                       dataset_copy[final_graph[k * i + j] * dim + d]) *
                      (dataset_copy[i * dim + d] -
                       dataset_copy[final_graph[k * i + j] * dim + d]);
        }
        dist_sum_lsh += sqrt(dist_lsh);
      }
    }
    std::cout << "overall dist after rand it: " << random_iteration
              << " dist: " << dist_sum_lsh << std::endl;
  }

  undo_randomize(k, dataset, final_graph, final_graph_distances, indices_map);

  std::chrono::time_point<std::chrono::system_clock> timer_stop =
      std::chrono::system_clock::now();
  last_duration_s =
      std::chrono::duration<double>(timer_stop - timer_start).count();
  return final_graph;
}

std::vector<int32_t>
OperationNearestNeighborSampled::knn_ocl(uint32_t k,
                                         std::string configFileName) {
  // sample
  // base::DataMatrix sampled_dataset = sample_dataset(1, dataset_count);
  base::DataMatrix sampled_dataset(dataset);

  // TODO: change to sample
  std::vector<int> graph(dataset_count * k, -1);
  auto operation_graph = std::unique_ptr<
      sgpp::datadriven::DensityOCLMultiPlatform::OperationCreateGraphOCL>(
      sgpp::datadriven::createNearestNeighborGraphConfigured(
          sampled_dataset, k, dim, configFileName));

  operation_graph->create_graph(graph);

  last_duration_s = operation_graph->getLastDuration();
  return graph;
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
