#include "sgpp/datadriven/operation/hash/OperationCreateGraphOCL/OperationCreateGraphOCL.hpp"

using namespace sgpp::datadriven::DensityOCLMultiPlatform;

int main() {
  size_t k = 2;

  size_t node_count = 5;
  std::vector<int> directed(node_count * k);
  std::fill(directed.begin(), directed.end(), -1);
  // one clusters: "0, 1, 2" and "3, 4" but "0, 1, 2" not connected to "3, 4"
  directed[0 * k + 0] = 1;
  directed[0 * k + 1] = 2;

  directed[1 * k + 0] = 0;
  directed[1 * k + 1] = 2;

  directed[2 * k + 0] = 0;
  directed[2 * k + 1] = 1;

  directed[3 * k + 0] = 2;
  directed[3 * k + 1] = 4;

  directed[4 * k + 0] = 2;
  directed[4 * k + 1] = 3;

  OperationCreateGraphOCL::neighborhood_list_t undirected =
      OperationCreateGraphOCL::make_undirected_graph(directed, k);

  for (size_t i = 0; i < node_count; i++) {
    for (size_t neighbor : undirected[i]) {
      std::cout << "n: " << i << " -> " << neighbor << std::endl;
    }
  }

  std::vector<int> node_cluster_map;
  std::vector<std::vector<int>> clusters;
  OperationCreateGraphOCL::get_clusters_from_undirected_graph(undirected, node_cluster_map,
                                                              clusters, 0);
  for (size_t i = 0; i < node_count; i++) {
    std::cout << "n: " << i << " belongs to cluster: " << node_cluster_map[i] << std::endl;
  }

  for (std::vector<int> &c : clusters) {
    std::cout << "member of cluster: ";
    bool first = true;
    for (int m : c) {
      if (first) {
        first = false;
      } else {
        std::cout << ", ";
      }
      std::cout << m;
    }
    std::cout << std::endl;
  }

  node_count = 8;
  directed = std::vector<int>(node_count * k);
  std::fill(directed.begin(), directed.end(), -1);
  // one clusters: "0, 1, 2" and "3, 4" but "0, 1, 2" not connected to "3, 4"
  directed[0 * k + 0] = 1;
  directed[0 * k + 1] = 2;

  directed[1 * k + 0] = 0;
  directed[1 * k + 1] = 2;

  directed[2 * k + 0] = 0;
  directed[2 * k + 1] = 1;

  directed[3 * k + 0] = -1;
  directed[3 * k + 1] = -1;

  directed[4 * k + 0] = 5;
  directed[4 * k + 1] = 6;

  directed[5 * k + 0] = 4;
  directed[5 * k + 1] = 6;

  directed[6 * k + 0] = 4;
  directed[6 * k + 1] = 5;

  directed[7 * k + 0] = -1;
  directed[7 * k + 1] = -1;

  undirected = OperationCreateGraphOCL::make_undirected_graph(directed, k);

  for (size_t i = 0; i < node_count; i++) {
    for (int neighbor : undirected[i]) {
      std::cout << "n: " << i << " -> " << neighbor << std::endl;
    }
    if (undirected[i].size() == 0) {
      std::cout << "n: " << i << " no neighbors" << std::endl;
    }
  }

  OperationCreateGraphOCL::get_clusters_from_undirected_graph(undirected, node_cluster_map,
                                                              clusters, 2);
  for (size_t i = 0; i < node_count; i++) {
    std::cout << "n: " << i << " belongs to cluster: " << node_cluster_map[i] << std::endl;
  }

  for (std::vector<int> &c : clusters) {
    std::cout << "member of cluster: ";
    bool first = true;
    for (int m : c) {
      if (first) {
        first = false;
      } else {
        std::cout << ", ";
      }
      std::cout << m;
    }
    std::cout << std::endl;
  }

  std::vector<size_t> reference_cluster_map =
      OperationCreateGraphOCL::find_clusters_recursive(directed, k);

  for (size_t i = 0; i < node_count; i++) {
    std::cout << "n: " << i << " belongs to cluster: " << reference_cluster_map[i] << std::endl;
  }
}
