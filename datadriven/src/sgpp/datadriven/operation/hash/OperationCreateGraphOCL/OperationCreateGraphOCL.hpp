// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#ifndef OPERATIONCREATEGRAPHOCL_H
#define OPERATIONCREATEGRAPHOCL_H

#include <omp.h>
#include <cassert>
#include <sgpp/base/exception/operation_exception.hpp>
#include <sgpp/base/grid/GridStorage.hpp>
#include <sgpp/base/opencl/OCLManager.hpp>
#include <sgpp/base/opencl/OCLOperationConfiguration.hpp>
#include <sgpp/base/operation/hash/OperationMultipleEval.hpp>
#include <sgpp/base/tools/SGppStopwatch.hpp>
#include <vector>
#include "KernelCreateGraph.hpp"

namespace sgpp {
namespace datadriven {
namespace DensityOCLMultiPlatform {

/// Pure virtual base class for the k nearest neighbor opencl operation
class OperationCreateGraphOCL {
 protected:
  double last_duration_create_graph;
  double acc_duration_create_graph;

  /// Recursive depth-first function for traversing the k nearest neighbor graph
  // parameters:
  // index index of data point for which to calculate its cluster
  // nodes the graph in a datasetsize * k format, k entries refer to the neighbors
  // cluster number of clusters -> index of the current cluster!
  // k size of the neighborhood
  // clusterList size of the cluster belonging to a specific data points???
  // overwrite ??? -> maybe to merge clusters?
  static size_t find_neighbors(size_t index, std::vector<int64_t> &nodes, size_t cluster, size_t k,
                               std::vector<size_t> &clusterList, bool overwrite = false) {
    clusterList[index] = cluster;  // assign current node to new cluster
    bool overwrite_enabled = overwrite;
    bool removed = true;
    // iterates the neighbors of the current node, k entries
    for (size_t i = index * k; i < (index + 1) * k; i++) {
      if (nodes[i] == -2) continue;  // endes with -2 have been pruned!
      removed = false;
      size_t currIndex = nodes[i];       // current neighbor
      if (nodes[currIndex * k] != -1) {  // does it exist?
        if (clusterList[currIndex] == 0 &&
            !overwrite_enabled) {            // has this node already been assigned?
          clusterList[currIndex] = cluster;  // assign neighbor to current cluster
                                             // recursively look at neighbors
          size_t ret_cluster =
              OperationCreateGraphOCL::find_neighbors(currIndex, nodes, cluster, k, clusterList);
          if (ret_cluster != cluster) {
            clusterList[index] = cluster;
            overwrite_enabled = true;
            i = index * k;
            continue;
          }
        } else if (!overwrite_enabled && clusterList[currIndex] != cluster) {
          // encountered a node that was already processed and belongs to a cluster
          cluster = clusterList[currIndex];
          clusterList[index] = cluster;
          overwrite_enabled = true;
          i = index * k - 1;
          continue;
        } else {
          // encountered another node that belongs to another cluster (connecting multiple prio
          // separate clusters)
          if (clusterList[currIndex] != cluster) {
            OperationCreateGraphOCL::find_neighbors(currIndex, nodes, cluster, k, clusterList,
                                                    true);
            clusterList[currIndex] = cluster;
          }
        }
      }
    }
    if (removed) {
      clusterList[index] = 0;
      cluster = 0;
    }
    return cluster;
  }

 public:
  OperationCreateGraphOCL() : last_duration_create_graph(0.0), acc_duration_create_graph(0.0) {}

  /// Pure virtual function to create the k nearest neighbor graph for some datapoints of a dataset
  virtual void create_graph(std::vector<int64_t> &resultVector, size_t startid = 0,
                            size_t chunksize = 0) = 0;
  virtual void create_graph(std::vector<int> &resultVector, size_t startid = 0,
                            size_t chunksize = 0) {
    std::vector<int64_t> graph_unconverted(resultVector.begin(), resultVector.end());
    create_graph(graph_unconverted, startid, chunksize);
    resultVector = std::vector<int>(graph_unconverted.begin(), graph_unconverted.end());
  }
  virtual void begin_graph_creation(size_t startid, size_t chunksize) = 0;
  virtual void finalize_graph_creation(std::vector<int64_t> &resultVector, size_t startid,
                                       size_t chunksize) = 0;
  /// Assign a clusterindex for each datapoint using the connected components of the graph
  // graph has k * #datapoints entries
  static std::vector<size_t> find_clusters_recursive(std::vector<int64_t> &graph, size_t k) {
    std::vector<size_t> clusters(graph.size() / k);
    size_t cluster_count = 0;
    std::fill(clusters.begin(), clusters.end(), 0);
    // for (size_t node = 0; node < clusters.size(); node++) {
    //   clusters[node] = 0;
    // }

    // check the cluster to which each data point belongs, i iterates data points
    for (size_t i = 0; i < clusters.size(); i++) {
      // check whether already assigned/whether data point is has any neighbors
      if (clusters[i] == 0 && graph[i * k] != -1) {
        // assume this is a new cluster
        cluster_count++;
        if (OperationCreateGraphOCL::find_neighbors(i, graph, cluster_count, k, clusters) !=
            cluster_count)
          cluster_count--;
      }
    }
    std::cout << "Found " << cluster_count << " clusters!" << std::endl;
    return clusters;
  }

  using neighborhood_list_t = std::vector<std::vector<int64_t>>;

  // directed format: index corresponds to node, k entries per index corresponding to connected
  // nodes (edges), -1 indicates empty
  static neighborhood_list_t make_undirected_graph(std::vector<int64_t> directed, size_t k) {
    size_t node_count = directed.size() / k;
    // neighborhood_list_t undirected(node_count, std::vector<int>());
    neighborhood_list_t undirected(node_count);
    // add directed edges
    // #pragma omp parallel for
    for (size_t i = 0; i < node_count; i += 1) {
      // std::cout << "node i: " << i << " neighbors: ";
      for (size_t cur_k = 0; cur_k < k; cur_k += 1) {
        // std::cout << directed[i * k + cur_k] << " ";
        if (directed[i * k + cur_k] == -1) {
          continue;
        }
        if (directed[i * k + cur_k] == -2) {
          // std::cout << "got a -2 for i: " << i << " cur_k: " << cur_k << std::endl;
          continue;
        }
        // assert that the index is in range
        // assert(directed[i * k + cur_k] < static_cast<int64_t>(node_count));
        // assert(directed[i * k + cur_k] >= 0);
        // if (directed[i * k + cur_k] > static_cast<int>(node_count)) {
        //   std::cerr << "error: index too large! i: " << i
        //             << " neighbor: " << directed[i * k + cur_k] << std::endl;
        //   continue;
        // }
        undirected[i].push_back(directed[i * k + cur_k]);
      }
      // std::cout << std::endl;
    }

    // for (size_t i = 0; i < node_count; i += 1) {
    //   std::cout << "uni node i: " << i << " count: " << undirected[i].size() << " neighbors:";
    //   for (int pointee : undirected[i]) {
    //     std::cout << " " << pointee;
    //   }
    //   std::cout << std::endl;
    // }

    // add reversed edge direction (if not already there)
    // std::vector<omp_lock_t> locks(node_count);
    // #pragma omp parallel for
    for (size_t i = 0; i < node_count; i += 1) {
      // not self-modifying: graph is non-reflexive
      // enumerate partners of i
      // std::cout << "i: " << i << std::endl;
      for (int64_t pointee : undirected[i]) {
        bool found = false;
        // std::cout << "pointee: " << pointee << std::endl;
        // std::cout << "members: " << undirected[pointee].size() << std::endl;
        for (int64_t candidate : undirected[pointee]) {
          if (candidate == static_cast<int64_t>(i)) {
            found = true;
            break;
          }
        }
        if (!found) {
          // omp_set_lock(&(locks[pointee]));
          // not in neighbor list -> add
          undirected[pointee].push_back(i);
          // omp_unset_lock(&(locks[pointee]));
        }
      }
    }
    return undirected;
  }

  static void get_clusters_from_undirected_graph(const neighborhood_list_t &undirected,
                                                 std::vector<int64_t> &node_cluster_map,
                                                 std::vector<std::vector<int64_t>> &clusters,
                                                 size_t cluster_size_min = 0) {
    // not yet processd set to -1, solitary set to -2
    node_cluster_map.resize(undirected.size());
    std::fill(node_cluster_map.begin(), node_cluster_map.end(), -1);
    clusters.clear();
    int64_t cluster_id = 0;
    for (size_t i = 0; i < undirected.size(); i++) {
      if (node_cluster_map[i] != -1) {
        // already processed
        continue;
      }
      // not yet processed, part of a new cluster (or solitary)
      std::vector<int64_t> cluster;
      cluster.push_back(i);
      size_t cur_node_index = 0;
      node_cluster_map[i] = cluster_id;
      while (cur_node_index < cluster.size()) {
        for (int64_t neighbor_index : undirected[cluster[cur_node_index]]) {
          if (node_cluster_map[neighbor_index] == -1) {
            // not yet processed, enqueue
            cluster.push_back(neighbor_index);
            node_cluster_map[neighbor_index] = cluster_id;
          } else {
            assert(node_cluster_map[neighbor_index] == cluster_id);
          }
        }
        cur_node_index += 1;
      }

      // now all cluster members are collected
      if (cluster.size() >= cluster_size_min) {
        clusters.push_back(std::move(cluster));
        cluster_id += 1;
      } else {
        // rejected, nodes are solitary
        for (int64_t member_index : cluster) {
          node_cluster_map[member_index] = -2;
        }
      }
    }
  }

  static void find_clusters(std::vector<int64_t> &directed_graph, size_t k,
                            std::vector<int64_t> &node_cluster_map, neighborhood_list_t &clusters) {
    // std::vector<int> node_cluster_map;
    // std::vector<std::vector<int>> clusters;

    OperationCreateGraphOCL::neighborhood_list_t undirected_graph =
        OperationCreateGraphOCL::make_undirected_graph(directed_graph, k);

    OperationCreateGraphOCL::get_clusters_from_undirected_graph(undirected_graph, node_cluster_map,
                                                                clusters, 2);
  }

  // TODO: incomplete
  // static void find_clusters_merging(std::vector<int64_t> &directed_graph, size_t k,
  //                                   std::vector<std::forward_list<int64_t>> &clusters_ret) {
  //   size_t dataset_size = directed_graph.size() / k;
  //   std::vector<int64_t> visited(dataset_size, -1);  // contains cluster_id
  //   std::vector<std::forward_list<int64_t>> partial_clusters;
  //   // index of vector is cluster id, inner vector is list of clusters to be added
  //   // to this cluster -> list empty, part of the final result
  //   std::vector<std::vector<int64_t>> merged_into;
  //   std::vector<std::vector<int64_t>> own_target;
  //   std::vector<std::forward_list<int64_t>> final_clusters;

  //   for (size_t dataset_index = 0; dataset_index < dataset_size; dataset_index += 1) {
  //     // already encountered by a previous iteration
  //     if (visited[dataset_index] != -1) {
  //       continue;
  //     }
  //     std::forward_list<int64_t> partial_cluster;
  //     partial_cluster.push_back(dataset_index);
  //     size_t cluster_id = partial_cluster.size();
  //     merged_into.push_back{};  // a new empty vector
  //     size_t rec_index = 0;
  //     int merge_target = -1;
  //     while (rec_index < partial_cluster.size()) {
  //       visited[rec_index] = cluster_id;
  //       // add all non-visited neighbors to recursion
  //       for (size_t j = 0; j < k; j += 1) {
  //         size_t neighbor_index = directed_graph[rec_index * k + j];
  //         size_t visited_state = visited[neighbor_index];
  //         if (visited_state == -1) {
  //           partial_cluster.push_back(neighbor_index);
  //         } else if (visited_state == cluster_id) {
  //           // already visited in this partial cluster run, nothing to do
  //         } else {
  //           // part of a different cluster, need to remember to merge
  //           if (merge_target == -1) {
  //             merge_target = visited_state;
  //           } else if (visited_state < merge_target) {
  //             merge_target = visited_state;
  //           }
  //         }
  //       }
  //     }

  //     if (merge_target != -1) {
  //       // - a merge needs to be done
  //       // - always merge into cluster with lower id
  //       // - every cluster specifies at most 1 merge
  //       if (visited_state < cluster_id) {
  //         merged_into[visited_state].push_back(cluster_id);
  //       } else {
  //         // implied ">", else cluster_id = max(all cluster_ids)
  //         // (this is only possible in the parallelized case)
  //         merged_into[cluster_id].push_back(visited_state);
  //       }
  //     }
  //     // all partial clusters have been created, now consolidate merge list
  //     for (size_t i = 0; i < partial_clusters.size() i += 1) {

  //     }

  //     partial_clusters.push_back(partial_cluster);
  //   }

  //   return final_clusters;
  // }

  virtual ~OperationCreateGraphOCL(void) {}

  /// Add the default parameters to the the configuration
  static void load_default_parameters(std::shared_ptr<base::OCLOperationConfiguration> parameters) {
    if (parameters->contains("INTERNAL_PRECISION") == false) {
      std::cout << "Warning! No internal precision setting detected."
                << " Using double precision from now on!" << std::endl;
      parameters->addIDAttr("INTERNAL_PRECISION", "double");
    }
    if ((*parameters)["INTERNAL_PRECISION"].get().compare("float") == 0) {
      DensityOCLMultiPlatform::KernelCreateGraph<float>::augmentDefaultParameters(*parameters);
    } else if ((*parameters)["INTERNAL_PRECISION"].get().compare("double") == 0) {
      DensityOCLMultiPlatform::KernelCreateGraph<double>::augmentDefaultParameters(*parameters);
    } else {
      std::stringstream errorString;
      errorString << "Error creating operation\"CreateGraphOCL\": "
                  << " invalid value for parameter \"INTERNAL_PRECISION\"";
      throw base::operation_exception(errorString.str().c_str());
    }
  }

  double getLastDuration() { return last_duration_create_graph; }

  void resetAccDuration() { acc_duration_create_graph = 0.0; }

  double getAccDuration() { return acc_duration_create_graph; }
};

}  // namespace DensityOCLMultiPlatform
}  // namespace datadriven
}  // namespace sgpp

#endif /* OPERATIONCREATEGRAPHOCL_H */
