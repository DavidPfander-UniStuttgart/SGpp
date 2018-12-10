#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <random>
#include <set>

#include "sgpp/datadriven/operation/hash/OperationCreateGraphOCL/ConnectedComponents.hpp"
#include "sgpp/datadriven/operation/hash/OperationCreateGraphOCL/DetectComponents.hpp"
#include "sgpp/datadriven/operation/hash/OperationCreateGraphOCL/DetectComponentsRecursive.hpp"

using high_resolution_clock = std::chrono::high_resolution_clock;
using time_point = std::chrono::high_resolution_clock::time_point;

namespace util {
void write_vector(const std::vector<int64_t> &v, const std::string &file_name) {
  std::ofstream ofs(file_name, std::ios::out | std::ofstream::binary);
  std::ostream_iterator<int64_t> osi{ofs, " "};  // " " separator
  std::copy(v.begin(), v.end(), osi);
}

std::vector<int64_t> read_vector(const std::string &file_name) {
  std::vector<int64_t> v;
  std::ifstream ifs(file_name, std::ios::in | std::ifstream::binary);
  std::istream_iterator<int64_t> iter(ifs);
  std::istream_iterator<int64_t> end;
  std::copy(iter, end, std::back_inserter(v));
  return v;
}

template <typename T>
struct random_int {
 private:
  std::random_device rd;  // to seed
  std::default_random_engine generator;
  std::uniform_int_distribution<int64_t> distribution;

 public:
  random_int(T left, T right) : generator(rd()), distribution(left, right) {}

  T operator()() { return distribution(generator); }
};

}  // namespace util

namespace clustering {

using neighborhood_list_t = std::vector<int64_t>;

neighborhood_list_t create_random_graph(size_t n_outer, size_t k_outer) {
  util::random_int<int64_t> r(0, n_outer - 1);

  neighborhood_list_t graph(n_outer * k_outer);
  for (size_t n = 0; n < n_outer; n += 1) {
    for (size_t k = 0; k < k_outer; k += 1) {
      graph[n * k_outer + k] = r();
    }
  }
  return graph;
}

void randomize_graph_consistently(neighborhood_list_t &graph, size_t k_outer) {
  const size_t N = graph.size() / k_outer;
  util::random_int<int64_t> r(0, N - 1);
  std::vector<size_t> origin_map(N);
  for (size_t i = 0; i < N; i += 1) {
    origin_map[i] = i;
  }
  // swap map and data
  for (size_t i = 0; i < N; i += 1) {
    size_t first_index = r();
    size_t second_index = r();
    std::swap(origin_map[first_index], origin_map[second_index]);
    for (size_t k = 0; k < k_outer; k += 1) {
      std::swap(graph[first_index * k_outer + k], graph[second_index * k_outer + k]);
    }
  }
  std::vector<size_t> map(N);
  for (size_t i = 0; i < N; i += 1) {
    map[origin_map[i]] = i;
  }
  // now rename everything, so that indices are consistent again
  for (size_t i = 0; i < N; i += 1) {
    for (size_t k = 0; k < k_outer; k += 1) {
      graph[i * k_outer + k] = map[graph[i * k_outer + k]];
    }
  }
}

// first neighbor is used to connect, others are randomized
neighborhood_list_t create_connect_component(size_t start_index, size_t cc_size, size_t k_outer) {
  util::random_int<int64_t> r(start_index, start_index + cc_size - 1);
  neighborhood_list_t cc(cc_size * k_outer);
  for (size_t n = 0; n < cc_size; n += 1) {
    if (n != cc_size - 1) {
      cc[n * k_outer + 0] = start_index + n + 1;
    } else {
      cc[n * k_outer + 0] = start_index;
    }
  }
  for (size_t n = 0; n < cc_size; n += 1) {
    for (size_t k = 1; k < k_outer; k += 1) {
      cc[n * k_outer + k] = r();
    }
  }
  return cc;
}

// disadvantage: always can reach whole cluster from every node of the cluster
neighborhood_list_t create_connected_graph(size_t num_cc, size_t cc_size, size_t k_outer,
                                           bool randomize = true) {
  neighborhood_list_t graph;
  size_t start_index = 0;
  for (size_t i = 0; i < num_cc; i += 1) {
    neighborhood_list_t cc = create_connect_component(start_index, cc_size, k_outer);
    graph.insert(graph.end(), cc.begin(), cc.end());
    start_index += cc_size;
  }

  if (randomize) {
    randomize_graph_consistently(graph, k_outer);
  }
  return graph;
}

void print_graph(const neighborhood_list_t graph, const size_t k) {
  const size_t N = graph.size() / k;
  for (size_t i = 0; i < N; i += 1) {
    for (size_t j = 0; j < k; j += 1) {
      if (j > 0) {
        std::cout << ", ";
      } else {
        std::cout << "i: " << i << " -> ";
      }
      std::cout << graph[i * k + j];
    }
    std::cout << std::endl;
  }
}

template <typename T>
void print_labeled(const std::vector<T> &labeled_datapoints) {
  for (size_t i = 0; i < labeled_datapoints.size(); i += 1) {
    std::cout << "i: " << i << " -> c_id: " << labeled_datapoints[i] << std::endl;
  }
}

void print_connected_components(std::vector<std::set<int64_t>> &all_clusters) {
  std::cout << "num_clusters: " << all_clusters.size() << std::endl;
  for (size_t i = 0; i < all_clusters.size(); i += 1) {
    auto &cluster_indices = all_clusters[i];
    bool first = true;
    for (const size_t &i : cluster_indices) {
      if (first) {
        first = false;
      } else {
        std::cout << ", ";
      }
      std::cout << i;
    }
    std::cout << std::endl;
  }
}

}  // namespace clustering

int main() {
  // constexpr size_t N = 100000;

  constexpr size_t num_cc = 100;
  constexpr size_t cc_size = 1000000;  // recursive can only do up to 10^4
  std::cout << "num datapoints: " << (num_cc * cc_size) << std::endl;

  constexpr size_t k = 2;

  // // merge two
  // constexpr size_t N = 4;
  // constexpr size_t k = 2;
  // clustering::neighborhood_list_t graph(N * k);
  // graph[0 * k + 0] = 0;
  // graph[0 * k + 1] = 0;
  // graph[1 * k + 0] = 0;
  // graph[1 * k + 1] = 0;
  // graph[2 * k + 0] = 2;
  // graph[2 * k + 1] = 2;
  // graph[3 * k + 0] = 1;
  // graph[3 * k + 1] = 2;

  // // merge four
  // constexpr size_t N = 7;
  // constexpr size_t k = 2;
  // clustering::neighborhood_list_t graph(N * k);
  // graph[0 * k + 0] = 0;
  // graph[0 * k + 1] = 0;
  // graph[1 * k + 0] = 1;
  // graph[1 * k + 1] = 1;
  // graph[2 * k + 0] = 2;
  // graph[2 * k + 1] = 2;
  // graph[3 * k + 0] = 3;
  // graph[3 * k + 1] = 3;
  // graph[4 * k + 0] = 5;
  // graph[4 * k + 1] = 6;
  // graph[5 * k + 0] = 0;
  // graph[5 * k + 1] = 1;
  // graph[6 * k + 0] = 2;
  // graph[6 * k + 1] = 3;

  // clustering::neighborhood_list_t graph = clustering::create_random_graph(N, k);
  clustering::neighborhood_list_t graph =
      clustering::create_connected_graph(num_cc, cc_size, k, true);
  // clustering::print_graph(graph, k);
  // util::write_vector(graph, "graph.bin");
  // clustering::neighborhood_list_t graph2 = util::read_vector("graph.bin");
  // bool is_equal = std::equal(graph.begin(), graph.end(), graph2.begin());
  // if (is_equal) {
  //   std::cout << "graphs are equal" << std::endl;
  // } else {
  //   std::cout << "graphs NOT equal" << std::endl;
  // }

  // std::vector<size_t> labeled_datapoints;
  // {
  //   time_point start = high_resolution_clock::now();
  //   labeled_datapoints = sgpp::datadriven::clustering::find_clusters_recursive(graph, k);
  //   time_point end = high_resolution_clock::now();
  //   double duration_s_avr = std::chrono::duration<double>(end - start).count();
  //   std::cout << "duration simple: " << duration_s_avr << std::endl;
  //   // std::cout << "recursive fcc:" << std::endl;
  //   // clustering::print_labeled(labeled_datapoints);
  // }

  // std::vector<size_t> labeled_datapoints2;
  // {
  //   std::vector<int64_t> labeled_datapoints_unconverted;
  //   std::vector<std::vector<int64_t>> clusters;
  //   time_point start = high_resolution_clock::now();
  //   // sgpp::datadriven::clustering::find_clusters(graph, k, labeled_datapoints_unconverted,
  //   // clusters);
  //   sgpp::datadriven::clustering::neighborhood_list_t undirected_graph =
  //       sgpp::datadriven::clustering::make_undirected_graph(graph, k);
  //   time_point end = high_resolution_clock::now();
  //   double duration_undirected_s = std::chrono::duration<double>(end - start).count();
  //   std::cout << "duration improved make_undirected_graph: " << duration_undirected_s <<
  //   std::endl; start = high_resolution_clock::now();
  //   sgpp::datadriven::clustering::get_clusters_from_undirected_graph(
  //       undirected_graph, labeled_datapoints_unconverted, clusters, 2);
  //   end = high_resolution_clock::now();
  //   double duration_detect_s = std::chrono::duration<double>(end - start).count();
  //   std::cout << "duration improved get_clusters_from_undirected_graph: " << duration_detect_s
  //             << std::endl;

  //   labeled_datapoints2.resize(labeled_datapoints_unconverted.size());
  //   for (size_t i = 0; i < labeled_datapoints_unconverted.size(); i += 1) {
  //     labeled_datapoints2[i] = static_cast<size_t>(labeled_datapoints_unconverted[i] + 1);
  //   }
  //   std::cout << "clusters improved: " << clusters.size() << std::endl;
  //   // std::cout << "duration improved: " << duration_s_avr << std::endl;
  //   // std::cout << "improved:" << std::endl;
  //   // clustering::print_labeled(labeled_datapoints2);
  // }

  // std::vector<size_t> labeled_datapoints;
  // {
  //   time_point start = high_resolution_clock::now();
  //   std::vector<std::set<int64_t>> all_clusters;
  //   sgpp::datadriven::clustering::connected_components(graph, k, labeled_datapoints,
  //   all_clusters);
  //   time_point end = high_resolution_clock::now();
  //   double duration_s_avr = std::chrono::duration<double>(end - start).count();
  //   std::cout << "duration new: " << duration_s_avr << std::endl;
  //   std::cout << "num_clusters: " << all_clusters.size() << std::endl;
  //   // clustering::print_connected_components(all_clusters);
  //   // std::cout << "cc map:" << std::endl;
  //   // clustering::print_labeled(labeled_datapoints);
  // }

  std::vector<int64_t> labeled_datapoints;
  {
    time_point start = high_resolution_clock::now();
    std::vector<std::vector<int64_t>> all_clusters;
    sgpp::datadriven::clustering::connected_components_no_set(graph, k, labeled_datapoints,
                                                              all_clusters);
    time_point end = high_resolution_clock::now();
    double duration_s_avr = std::chrono::duration<double>(end - start).count();
    std::cout << "duration new: " << duration_s_avr << std::endl;
    std::cout << "num_clusters: " << all_clusters.size() << std::endl;
    // clustering::print_connected_components(all_clusters);
    // std::cout << "cc map:" << std::endl;
    // clustering::print_labeled(labeled_datapoints);
  }

  // bool is_equal =
  //     std::equal(labeled_datapoints.begin(), labeled_datapoints.end(),
  //     labeled_datapoints2.begin());
  // if (is_equal) {
  //   std::cout << "graphs are equal" << std::endl;
  // } else {
  //   std::cout << "graphs NOT equal" << std::endl;
  // }

  // {
  //   // std::vector<std::vector<int64_t>> clusters;
  //   std::vector<int64_t> labeled_datapoints = clustering::find_clusters_distributed(graph, k);
  //   std::cout << "distributed:" << std::endl;
  //   clustering::print_labeled(labeled_datapoints);
  // }

  return EXIT_SUCCESS;
}
