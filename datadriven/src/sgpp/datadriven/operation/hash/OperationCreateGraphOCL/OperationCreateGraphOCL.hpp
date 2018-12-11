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
#include "DetectComponents.hpp"
#include "DetectComponentsRecursive.hpp"
#include "KernelCreateGraph.hpp"

namespace sgpp {
namespace datadriven {
namespace DensityOCLMultiPlatform {

/// Pure virtual base class for the k nearest neighbor opencl operation
class OperationCreateGraphOCL {
 protected:
  double last_duration_create_graph;
  double acc_duration_create_graph;

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
