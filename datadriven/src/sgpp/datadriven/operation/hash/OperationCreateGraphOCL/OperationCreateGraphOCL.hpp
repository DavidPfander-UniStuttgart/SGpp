// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#ifndef OPERATIONCREATEGRAPHOCL_H
#define OPERATIONCREATEGRAPHOCL_H

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
  /// Recursive depth-first function for traversing the k nearest neighbor graph
  // parameters:
  // index index of data point for which to calculate its cluster
  // nodes the graph in a datasetsize * k format, k entries refer to the neighbors
  // cluster number of clusters -> index of the current cluster!
  // k size of the neighborhood
  // clusterList size of the cluster belonging to a specific data points???
  // overwrite ??? -> maybe to merge clusters?
  static size_t find_neighbors(size_t index, std::vector<int> &nodes, size_t cluster, size_t k,
                               std::vector<size_t> &clusterList, bool overwrite = false) {
    clusterList[index] = cluster;  // assign current node to new cluster
    bool overwrite_enabled = overwrite;
    bool removed = true;
    // iterates the neighbors of the current node, k entries
    for (size_t i = index * k; i < (index + 1) * k; i++) {
      if (nodes[i] == -2) continue;
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
            std::cout << "in ret_cluster != cluster" << std::endl;
            clusterList[index] = cluster;
            overwrite_enabled = true;
            i = index * k;
            continue;
          }
        } else if (!overwrite_enabled &&
                   clusterList[currIndex] != cluster) {  // part of other cluster?
          // encountered a node that was already processed and belongs to a cluster
          std::cout << "in clusterList[currIndex] != cluster" << std::endl;
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
  OperationCreateGraphOCL() {}

  /// Pure virtual function to create the k nearest neighbor graph for some datapoints of a dataset
  virtual void create_graph(std::vector<int> &resultVector, int startid = 0, int chunksize = 0) = 0;
  virtual void begin_graph_creation(int startid, int chunksize) = 0;
  virtual void finalize_graph_creation(std::vector<int> &resultVector, int startid,
                                       int chunksize) = 0;
  /// Assign a clusterindex for each datapoint using the connected components of the graph
  // graph has k * #datapoints entries
  static std::vector<size_t> find_clusters(std::vector<int> &graph, size_t k) {
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

  using neighborhood_list_t = std::vector<std::vector<size_t>>;

  // directed format: index corresponds to node, k entries per index corresponding to connected
  // nodes (edges), -1 indicates empty
  static neighborhood_list_t make_undirected_graph(std::vector<int> directed, size_t k) {
    size_t node_count = directed.size() / k;
    neighborhood_list_t undirected(node_count);
    // add directed edges
    for (size_t i = 0; i < node_count; i += 1) {
      for (size_t cur_k = 0; cur_k < k; cur_k += 1) {
        if (directed[i * k + cur_k] == -1) {
          break;
        }
        undirected[i].push_back(directed[i * k + cur_k]);
      }
    }
    // add reversed edge direction (if not already there)
    for (size_t i = 0; i < node_count; i += 1) {
      // not self-modifying: graph is non-reflexive
      for (size_t pointee : undirected[i]) {
        bool found = false;
        for (size_t candidate : undirected[pointee]) {
          if (candidate == i) {
            found = true;
            break;
          }
        }
        if (!found) {
          // not in neighbor list -> add
          undirected[pointee].push_back(i);
        }
      }
    }
    return undirected;
  }

  static void get_clusters_from_undirected_graph(const neighborhood_list_t &undirected,
                                                 std::vector<int> &node_cluster_map,
                                                 std::vector<std::vector<size_t>> &clusters,
                                                 size_t cluster_size_min = 0) {
    // not yet processd set to -1, solitary set to -2
    node_cluster_map.resize(undirected.size());
    std::fill(node_cluster_map.begin(), node_cluster_map.end(), -1);
    clusters.clear();
    int cluster_id = 0;
    for (size_t i = 0; i < undirected.size(); i++) {
      if (node_cluster_map[i] != -1) {
        // already processed
        continue;
      }
      // not yet processed, part of a new cluster (or solitary)
      std::vector<size_t> cluster;
      cluster.push_back(i);
      size_t cur_node_index = 0;
      node_cluster_map[i] = cluster_id;
      while (cur_node_index < cluster.size()) {
        for (size_t neighbor_index : undirected[cluster[cur_node_index]]) {
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
        for (size_t member_index : cluster) {
          node_cluster_map[member_index] = -2;
        }
      }
    }
  }

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
};

}  // namespace DensityOCLMultiPlatform
}  // namespace datadriven
}  // namespace sgpp

#endif /* OPERATIONCREATEGRAPHOCL_H */
