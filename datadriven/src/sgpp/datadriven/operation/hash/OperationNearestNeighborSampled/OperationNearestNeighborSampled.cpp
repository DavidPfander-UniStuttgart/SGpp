#include "OperationNearestNeighborSampled.hpp"
#include "KNNFactory.hpp"
#include "sgpp/datadriven/operation/hash/OperationCreateGraphOCL/OpFactory.hpp"
#include "sgpp/globaldef.hpp"

namespace sgpp {
namespace datadriven {

OperationNearestNeighborSampled::OperationNearestNeighborSampled(
    base::DataMatrix dataset, size_t dim)
    : dataset(dataset), dataset_count(dataset.getNrows()), dim(dim) {}

std::vector<int32_t>
OperationNearestNeighborSampled::kNN_lsh(uint32_t k, uint64_t lsh_tables,
                                         uint64_t lsh_hashes, double lsh_w) {
  // sample

  // transpose
  dataset.transpose(); // TODO: change to sample
  std::unique_ptr<lshknn::KNN> lsh(lshknn::create_knn_lsh(
      dataset, dataset_count, dim, lsh_tables, lsh_hashes, lsh_w));
  std::chrono::time_point<std::chrono::system_clock> timer_lsh_start =
      std::chrono::system_clock::now();
  std::vector<int> graph = lsh->kNearestNeighbors(k);
  std::chrono::time_point<std::chrono::system_clock> timer_lsh_stop =
      std::chrono::system_clock::now();
  last_duration_s =
      std::chrono::duration<double>(timer_lsh_stop - timer_lsh_start).count();

  return graph;
}

std::vector<int32_t>
OperationNearestNeighborSampled::kNN_naive(uint32_t k,
                                           std::string configFileName) {
  // sample

  // TODO: change to sample
  std::vector<int> graph(dataset_count * k, -1);
  auto operation_graph = std::unique_ptr<
      sgpp::datadriven::DensityOCLMultiPlatform::OperationCreateGraphOCL>(
      sgpp::datadriven::createNearestNeighborGraphConfigured(dataset, k, dim,
                                                             configFileName));

  operation_graph->create_graph(graph);

  last_duration_s = operation_graph->getLastDuration();
  return graph;
}

} // namespace datadriven
} // namespace sgpp
