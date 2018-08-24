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

std::vector<int32_t> OperationNearestNeighborSampled::knn_naive(uint32_t k) {
  return knn_naive(dataset, k);
}

std::vector<int32_t>
OperationNearestNeighborSampled::knn_naive(base::DataMatrix &local_dataset,
                                           uint32_t k) {
  // sample
  base::DataMatrix sampled_dataset(local_dataset);
  // base::DataMatrix sampled_dataset = sample_dataset(chunk_size, chunk_count);

  // transpose
  sampled_dataset.transpose();
  std::unique_ptr<lshknn::KNN> lsh(lshknn::create_knn_naive_cpu(
      sampled_dataset, sampled_dataset.getNcols(), dim));
  std::chrono::time_point<std::chrono::system_clock> timer_lsh_start =
      std::chrono::system_clock::now();
  std::vector<int> graph = lsh->kNearestNeighbors(k);
  std::chrono::time_point<std::chrono::system_clock> timer_lsh_stop =
      std::chrono::system_clock::now();
  last_duration_s =
      std::chrono::duration<double>(timer_lsh_stop - timer_lsh_start).count();
  return graph;
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
    size_t k, std::vector<int> &graph, std::vector<double> &graph_distances,
    size_t first_index) { // size_t &far_neighbor_index, double
                          // &far_neighbor_distance

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
    size_t k, base::DataMatrix &dataset, size_t chunk_first_index,
    size_t chunk_range, std::vector<int> &final_graph,
    std::vector<double> &final_graph_distances, std::vector<int> &partial_graph,
    std::vector<int> &indices_map) {
  for (size_t i = 0; i < chunk_range; i += 1) {
    // next index in graph to process (updating its neighbors)
    // first look up which index in the final graph is to be updated
    // dataset index in the randomized data
    const size_t mapped_index = chunk_first_index + i;
    // original index, needed to properly access the final graph
    const int original_index = indices_map[mapped_index];
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
        far_neighbor(k, final_graph, final_graph_distances, original_index);
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
      size_t mapped_candidate_index =
          chunk_first_index + partial_graph[i * k + j];
      int unmapped_candidate_index = indices_map[mapped_candidate_index];
      // test that the data point used actually matches
      for (size_t dd = 0; dd < dim; dd += 1) {
        assert(dataset[mapped_candidate_index * dim + dd] ==
               dataset_copy[unmapped_candidate_index * dim + dd]);
        // std::cout << "map: " << mapped_candidate_index
        //           << " unmapped: " << unmapped_candidate_index
        //           << " d map: " << dataset[mapped_candidate_index * dim + dd]
        //           << " d unmapped: "
        //           << dataset_copy[unmapped_candidate_index * dim + dd]
        //           << std::endl;
      }
      double dist = l2_dist(dataset, mapped_index, mapped_candidate_index);
      // std::cout << "dist: " << dist << " max_dist: " << max_dist << std::endl;
      if (dist < max_dist) {
        // std::cout << "update!" << std::endl;

        // std::cout << "partial_graph[" << mapped_chunk_index << " * " << k
        //           << " + " << j
        //           << "] = " << partial_graph[mapped_chunk_index * k + j]
        //           << " -- unmapped to --> " << unmapped_candidate_index
        //           << std::endl;
        bool is_duplicate = false;
        for (size_t h = 0; h < k; h += 1) {
          if (final_graph[original_index * k + h] == unmapped_candidate_index) {
            is_duplicate = true;
            break;
          }
        }
        if (is_duplicate) {
          // std::cout << "is duplicate!" << std::endl;
          continue;
        }
        assert(final_graph[original_index * k + max_dist_index] !=
               unmapped_candidate_index);
        // std::cout << "non-duplicate update!" << std::endl;
        final_graph_distances[original_index * k + max_dist_index] = dist;
        final_graph[original_index * k + max_dist_index] =
            unmapped_candidate_index;
        if (j != k - 1) {
          std::tie(max_dist_index, max_dist) = far_neighbor(
              k, final_graph, final_graph_distances, original_index);
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

void OperationNearestNeighborSampled::randomize(base::DataMatrix &dataset,
                                                std::vector<int> &indices_map) {
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<size_t> dist(0, dataset.getNrows() - 1);
  for (size_t i = 0; i < dataset.getNrows(); i += 1) {
    size_t swap_index = dist(mt);
    for (size_t d = 0; d < dim; d += 1) {
      double temp = dataset[swap_index * dim + d];
      dataset[swap_index * dim + d] = dataset[i * dim + d];
      dataset[i * dim + d] = temp;
    }
    int index_temp = indices_map[swap_index];
    indices_map[swap_index] = indices_map[i];
    indices_map[i] = index_temp;
  }
}

base::DataMatrix
OperationNearestNeighborSampled::undo_randomize(base::DataMatrix &dataset,
                                                std::vector<int> &indices_map) {
  base::DataMatrix original(dataset_count, dim);
  for (size_t i = 0; i < dataset.getNrows(); i += 1) {
    int cur = indices_map[i];
    for (size_t d = 0; d < dim; d += 1) {
      original[cur * dim + d] = dataset[i * dim + d];
    }
  }
  return original;
}

std::vector<int32_t> OperationNearestNeighborSampled::knn_naive_sampling(
    uint32_t k, uint32_t input_chunk_size, uint32_t randomize_count) {
  std::chrono::time_point<std::chrono::system_clock> timer_start =
      std::chrono::system_clock::now();

  size_t total_input_chunks = dataset_count / input_chunk_size;
  if (total_input_chunks * input_chunk_size < dataset_count) {
    total_input_chunks += 1;
  }
  std::cout << "total_input_chunks: " << total_input_chunks << std::endl;

  std::vector<int> indices_map(dataset_count);
  for (size_t i = 0; i < dataset_count; i += 1) {
    indices_map[i] = i;
  }

  randomize(dataset, indices_map);

  std::vector<int> final_graph(dataset_count * k, -1);
  std::vector<double> final_graph_distances(dataset_count * k);

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
      base::DataMatrix chunk(chunk_range, dim); // TODO: padding!
      for (size_t i = 0; i < chunk_range; i += 1) {
        for (size_t d = 0; d < dim; d += 1) {
          chunk[i * dim + d] = dataset[(chunk_first_index + i) * dim + d];
        }
      }

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
      std::vector<int> partial_graph = knn_naive(chunk, k);

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

      merge_knn(k, dataset, chunk_first_index, chunk_range, final_graph,
                final_graph_distances, partial_graph, indices_map);
    }
    randomize(dataset, indices_map);

    double dist_sum_lsh = 0.0;
    for (size_t i = 0; i < dataset_count; i += 1) {
      for (int j = 0; j < k; ++j) {
        double dist_lsh = 0.0;
        for (int d = 0; d < dim; ++d) {
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

  dataset = undo_randomize(dataset, indices_map);

  for (size_t i = 0; i < dataset_count; i += 1) {
    for (size_t d = 0; d < dim; d += 1) {
      assert(dataset[i * dim + d] == dataset_copy[i * dim + d]);
    }
  }

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

double
OperationNearestNeighborSampled::test_accuracy(const std::vector<int> correct,
                                               const std::vector<int> result,
                                               const int size, const int k) {
  int count = 0;
  int total = 0;
  for (int s = 0; s < size; s += 1) {
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
    const std::vector<double> data, const std::vector<int> correct,
    const std::vector<int> result, const int size, const int dim, const int k) {
  double dist_sum_correct = 0.0;
  double dist_sum_lsh = 0.0;
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < k; ++j) {
      double dist_correct = 0.0;
      for (int d = 0; d < dim; ++d) {
        dist_correct +=
            (data[i * dim + d] - data[correct[k * i + j] * dim + d]) *
            (data[i * dim + d] - data[correct[k * i + j] * dim + d]);
      }
      dist_sum_correct += sqrt(dist_correct);
    }
    for (int j = 0; j < k; ++j) {
      double dist_lsh = 0.0;
      for (int d = 0; d < dim; ++d) {
        dist_lsh += (data[i * dim + d] - data[result[k * i + j] * dim + d]) *
                    (data[i * dim + d] - data[result[k * i + j] * dim + d]);
      }
      dist_sum_lsh += sqrt(dist_lsh);
    }
  }
  std::cout << "dist_sum_correct: " << dist_sum_correct << std::endl;
  std::cout << "dist_sum_lsh: " << dist_sum_lsh << std::endl;
  return std::abs(dist_sum_lsh - dist_sum_correct) / dist_sum_correct;
}

std::vector<int>
OperationNearestNeighborSampled::read_csv(const std::string &csv_file_name) {
  std::vector<int> neighbors_csv;
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
    const std::string &graph_file_name, std::vector<int> &graph, uint64_t k) {
  std::ofstream out_graph(graph_file_name);
  for (size_t i = 0; i < graph.size() / k; ++i) {
    bool first = true;
    for (size_t j = 0; j < k; ++j) {
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
