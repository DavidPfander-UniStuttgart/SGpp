// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#ifdef __AVX__

#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <zlib.h>

#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "sgpp/base/grid/generation/functors/SurplusRefinementFunctor.hpp"
#include "sgpp/base/operation/BaseOpFactory.hpp"
#include "sgpp/base/operation/hash/OperationMultipleEval.hpp"
#include "sgpp/base/tools/ConfigurationParameters.hpp"
#include "sgpp/datadriven/DatadrivenOpFactory.hpp"
#include "sgpp/datadriven/operation/hash/OperationNearestNeighborSampled/OperationNearestNeighborSampled.hpp"
#include "sgpp/datadriven/tools/ARFFTools.hpp"
#include "sgpp/globaldef.hpp"
#include "test_datadrivenCommon.hpp"

struct StaticDataFixture {
  std::string dataset_file = "datasets/gaussian_c3_size20_dim2.arff";
  std::string reference_neighbors_file =
      "datasets/gaussian_c3_size20_dim2_reference.arff";
  sgpp::datadriven::Dataset dataset;
  uint32_t dim;
  uint64_t dataset_count;
  sgpp::base::DataMatrix trainingData;
  uint32_t k{5};
  uint32_t lsh_tables{50};
  uint32_t lsh_hashes{15};
  double lsh_w{1.0};
  // TODO: fix if used
  std::string config_file{"config_ocl_float_Quadro100.cfg"};
  uint32_t sampling_chunk_size{10};
  uint32_t sampling_randomize{3};

  StaticDataFixture() {
    dataset = sgpp::datadriven::ARFFTools::readARFF(dataset_file);
    dim = dataset.getDimension();
    trainingData = dataset.getData();
    dataset_count = trainingData.getNrows();
  }
  ~StaticDataFixture() {}
};

BOOST_FIXTURE_TEST_SUITE(NearestNeighborSampled, StaticDataFixture)

BOOST_AUTO_TEST_CASE(knn_naive) {

  sgpp::datadriven::OperationNearestNeighborSampled knn_op(true);
  std::vector<int64_t> neighbors_reference =
      knn_op.read_csv(reference_neighbors_file);
  std::vector<int64_t> graph = knn_op.knn_naive(dim, trainingData, k);
  double err_acc = knn_op.test_accuracy(neighbors_reference, graph,
                                        trainingData.getNrows(), k);
  double err_distance_acc =
      knn_op.test_distance_accuracy(trainingData, neighbors_reference, graph,
                                    trainingData.getNrows(), dim, k);
  BOOST_CHECK(err_acc == 1.0);
  BOOST_CHECK(err_distance_acc < 1E-10);
}

BOOST_AUTO_TEST_CASE(randomize_and_undo) {

  sgpp::base::DataMatrix dataset_copy(trainingData);

  sgpp::datadriven::OperationNearestNeighborSampled knn_op(true);
  std::vector<int64_t> graph_dummy(dataset_count * k);
  std::vector<double> graph_distance_dummy(dataset_count * k);
  std::vector<int64_t> indices_map(dataset_count);
  size_t filler_index = 0;
  std::generate(indices_map.begin(), indices_map.end(),
                [&filler_index]() { return filler_index++; });
  filler_index = 0;
  std::generate(graph_dummy.begin(), graph_dummy.end(),
                [&filler_index]() { return filler_index++; });
  double filler_double = 0.0;
  std::generate(graph_distance_dummy.begin(), graph_distance_dummy.end(),
                [&filler_double]() {
                  double ret = filler_double;
                  filler_double += 1;
                  return ret;
                });
  std::vector<int64_t> graph_dummy_copy(graph_dummy);
  std::vector<double> graph_distance_dummy_copy(graph_distance_dummy);

  // std::for_each(indices_map.begin(), indices_map.end(),
  //               [](int64_t i) { std::cout << i << " " << std::endl; });

  for (size_t i = 0; i < 10; i += 1) {
    knn_op.randomize(dim, trainingData, k, graph_dummy, graph_distance_dummy,
                     indices_map);
  }
  knn_op.undo_randomize(dim, trainingData, k, graph_dummy, graph_distance_dummy,
                        indices_map);
  // for (size_t i = 0; i < dataset_count; i += 1) {
  //   std::cout << "i: " << i << " ";
  //   for (size_t d = 0; d < dim; d += 1) {
  //     if (d > 0) {
  //       std::cout << ", ";
  //     }
  //     if (trainingData[i * dim + d] == dataset_copy[i * dim + d]) {
  //       std::cout << "(" << trainingData[i * dim + d]
  //                 << "==" << dataset_copy[i * dim + d] << ")";
  //     } else {
  //       std::cout << "(" << trainingData[i * dim + d]
  //                 << "!=" << dataset_copy[i * dim + d] << ")";
  //     }
  //   }
  //   std::cout << std::endl;
  // }
  for (size_t i = 0; i < dataset_count; i += 1) {
    for (size_t d = 0; d < dim; d += 1) {
      BOOST_CHECK(trainingData[i * dim + d] == dataset_copy[i * dim + d]);
    }
  }
  for (size_t i = 0; i < dataset_count; i += 1) {
    for (size_t j = 0; j < k; j += 1) {
      BOOST_CHECK(graph_dummy[i * k + j] == graph_dummy_copy[i * k + j]);
    }
  }
  for (size_t i = 0; i < dataset_count; i += 1) {
    for (size_t j = 0; j < k; j += 1) {
      BOOST_CHECK(graph_distance_dummy[i * k + j] ==
                  graph_distance_dummy_copy[i * k + j]);
    }
  }
  for (size_t i = 0; i < dataset_count; i += 1) {
    BOOST_CHECK(indices_map[i] == -1);
  }
}

BOOST_AUTO_TEST_CASE(knn_naive_sampling) {

  sampling_randomize = 3;
  sampling_chunk_size = 20;

  sgpp::base::DataMatrix trainingData_copy(trainingData);
  sgpp::datadriven::OperationNearestNeighborSampled knn_op(true);
  std::vector<int64_t> neighbors_reference =
      knn_op.read_csv(reference_neighbors_file);
  std::vector<int64_t> graph = knn_op.knn_naive_sampling(
      dim, trainingData, k, sampling_chunk_size, sampling_randomize);

  for (size_t i = 0; i < dataset_count; i += 1) {
    for (size_t d = 0; d < dim; d += 1) {
      BOOST_CHECK(trainingData[i * dim + d] == trainingData_copy[i * dim + d]);
    }
  }

  double err_acc = knn_op.test_accuracy(neighbors_reference, graph,
                                        trainingData.getNrows(), k);
  double err_distance_acc =
      knn_op.test_distance_accuracy(trainingData, neighbors_reference, graph,
                                    trainingData.getNrows(), dim, k);
  std::cout << "err_acc: " << err_acc << std::endl;
  std::cout << "err_distance_acc: " << err_distance_acc << std::endl;
  BOOST_CHECK(err_acc > 0.9);
  BOOST_CHECK(err_distance_acc < 1E-10);
}

BOOST_AUTO_TEST_CASE(knn_naive_sampling_one_chunk) {

  sampling_randomize = 1;
  sampling_chunk_size = 20;

  sgpp::base::DataMatrix trainingData_copy(trainingData);
  sgpp::datadriven::OperationNearestNeighborSampled knn_op(true);
  std::vector<int64_t> neighbors_reference =
      knn_op.read_csv(reference_neighbors_file);
  std::vector<int64_t> graph = knn_op.knn_naive_sampling(
      dim, trainingData, k, sampling_chunk_size, sampling_randomize);

  for (size_t i = 0; i < dataset_count; i += 1) {
    for (size_t d = 0; d < dim; d += 1) {
      BOOST_CHECK(trainingData[i * dim + d] == trainingData_copy[i * dim + d]);
    }
  }

  double err_acc = knn_op.test_accuracy(neighbors_reference, graph,
                                        trainingData.getNrows(), k);
  double err_distance_acc =
      knn_op.test_distance_accuracy(trainingData, neighbors_reference, graph,
                                    trainingData.getNrows(), dim, k);
  BOOST_CHECK(err_acc == 1.0);
  BOOST_CHECK(err_distance_acc < 1E-10);
}

BOOST_AUTO_TEST_CASE(merge_knn_minus_one) {

  dim = 2;
  k = 2;
  dataset_count = 4;
  size_t chunk_range = 4;
  sgpp::base::DataMatrix chunk(dataset_count, dim);
  for (size_t i = 0; i < dataset_count; i += 1) {
    for (size_t d = 0; d < dim; d += 1) {
      chunk[i * dim + d] = 0.5;
    }
  }

  std::vector<int64_t> partial_final_graph{
      -1, -1, //
      -1, -1, //
      -1, -1, //
      -1, -1, //

  };
  std::vector<double> partial_final_graph_distances{
      4, 4, //
      4, 4, //
      4, 4, //
      4, 4, //

  };
  std::vector<int64_t> partial_graph{
      0, 1, //
      0, 1, //
      0, 1, //
      0, 1, //
  };

  std::vector<int64_t> indices_map{0, 1, 2, 3};

  sgpp::datadriven::OperationNearestNeighborSampled knn_op(true);

  for (size_t chunk_first_index = 0; chunk_first_index < dataset_count;
       chunk_first_index += chunk_range) {
    knn_op.merge_knn(dim, k, chunk, chunk_range, partial_final_graph,
                     partial_final_graph_distances, partial_graph, indices_map);
    // for (size_t i = chunk_first_index; i < chunk_first_index + chunk_range;
    //      i += 1) {
    // for (size_t i = 0; i < chunk_range; i += 1) {
    //   for (size_t j = 0; j < k; j += 1) {
    //     if (j > 0) {
    //       std::cout << ", ";
    //     }
    //     std::cout << partial_final_graph[i * k + j];
    //   }
    //   std::cout << std::endl;
    // }
    for (size_t i = 0; i < chunk_range; i += 1) {
      for (size_t j = 0; j < k; j += 1) {
        if (j == 0) {
          BOOST_CHECK(partial_final_graph[i * k + j] == 0);
        } else {
          BOOST_CHECK(partial_final_graph[i * k + j] == 1);
        }
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(merge_knn_second_round_single_range) {

  dim = 2;
  k = 2;
  dataset_count = 4;
  size_t chunk_range = 4;
  sgpp::base::DataMatrix chunk(dataset_count, dim);
  // first data tuple
  chunk[0] = 1.0;
  chunk[1] = 1.0;
  // second data tuple
  chunk[3] = 1.0;
  chunk[4] = 1.0;
  for (size_t i = 2; i < dataset_count; i += 1) {
    for (size_t d = 0; d < dim; d += 1) {
      chunk[i * dim + d] = 0.5;
    }
  }

  std::vector<int64_t> partial_final_graph{
      0, 1, //
      0, 1, //
      0, 1, //
      0, 1  //
  };
  constexpr double init = 0.5 * sqrt(2);
  std::vector<double> partial_final_graph_distances{
      0.0,  0.0,  //
      0.0,  0.0,  //
      init, init, //
      init, init  //
  };
  std::vector<int64_t> partial_graph{
      2, 3, //
      2, 3, //
      2, 3, //
      2, 3, //
  };

  std::vector<int64_t> indices_map{0, 1, 2, 3};

  for (size_t chunk_first_index = 0; chunk_first_index < dataset_count;
       chunk_first_index += chunk_range) {

    sgpp::datadriven::OperationNearestNeighborSampled knn_op(true);
    knn_op.merge_knn(dim, k, chunk, chunk_range, partial_final_graph,
                     partial_final_graph_distances, partial_graph, indices_map);
    BOOST_CHECK(partial_final_graph[0] == 0);
    BOOST_CHECK(partial_final_graph[1] == 1);
    BOOST_CHECK(partial_final_graph[2] == 0);
    BOOST_CHECK(partial_final_graph[3] == 1);
    BOOST_CHECK(partial_final_graph[4] == 2);
    BOOST_CHECK(partial_final_graph[5] == 3);
    BOOST_CHECK(partial_final_graph[6] == 2);
    BOOST_CHECK(partial_final_graph[7] == 3);
    BOOST_CHECK(partial_final_graph_distances[0] == 0.0);
    BOOST_CHECK(partial_final_graph_distances[1] == 0.0);
    BOOST_CHECK(partial_final_graph_distances[2] == 0.0);
    BOOST_CHECK(partial_final_graph_distances[3] == 0.0);
    BOOST_CHECK(partial_final_graph_distances[4] == 0.0);
    BOOST_CHECK(partial_final_graph_distances[5] == 0.0);
    BOOST_CHECK(partial_final_graph_distances[6] == 0.0);
    BOOST_CHECK(partial_final_graph_distances[7] == 0.0);
  }
}

BOOST_AUTO_TEST_CASE(extract_chunk) {
  dim = 2;
  k = 2;
  dataset_count = 4;
  size_t chunk_range = 2;

  sgpp::base::DataMatrix trainingData(dataset_count, dim);
  trainingData[0] = 1.0;
  trainingData[1] = 2.0;
  trainingData[2] = 3.0;
  trainingData[3] = 4.0;
  trainingData[4] = 5.0;
  trainingData[5] = 6.0;
  trainingData[6] = 7.0;
  trainingData[7] = 8.0;
  sgpp::datadriven::OperationNearestNeighborSampled knn_op(true);

  std::vector<int64_t> final_graph{
      1, 2, //
      2, 3, //
      3, 4, //
      4, 5  //
  };
  std::vector<double> final_graph_distances{
      0.1, 0.2, //
      0.3, 0.4, //
      0.5, 0.6, //
      0.7, 0.8  //
  };
  std::vector<int64_t> indices_map{0, 1, 2, 3};

  for (size_t chunk_first_index = 0; chunk_first_index < dataset_count;
       chunk_first_index += chunk_range) {
    std::cout << "chunk_first_index: " << chunk_first_index << std::endl;
    auto [chunk, partial_final_graph, partial_final_graph_distances,
          partial_indices_map] =
        knn_op.extract_chunk(dim, k, chunk_first_index, chunk_range,
                             trainingData, final_graph, final_graph_distances,
                             indices_map);
    if (chunk_first_index == 0) {
      BOOST_CHECK(chunk[0] == trainingData[0]);
      BOOST_CHECK(chunk[1] == trainingData[1]);
      BOOST_CHECK(chunk[2] == trainingData[2]);
      BOOST_CHECK(chunk[3] == trainingData[3]);

      BOOST_CHECK(partial_final_graph[0] == final_graph[0]);
      BOOST_CHECK(partial_final_graph[1] == final_graph[1]);
      BOOST_CHECK(partial_final_graph[2] == final_graph[2]);
      BOOST_CHECK(partial_final_graph[3] == final_graph[3]);

      BOOST_CHECK(partial_final_graph_distances[0] == final_graph_distances[0]);
      BOOST_CHECK(partial_final_graph_distances[1] == final_graph_distances[1]);
      BOOST_CHECK(partial_final_graph_distances[2] == final_graph_distances[2]);
      BOOST_CHECK(partial_final_graph_distances[3] == final_graph_distances[3]);

      BOOST_CHECK(partial_indices_map[0] == 0);
      BOOST_CHECK(partial_indices_map[1] == 1);
    } else if (chunk_first_index == 2) {
      BOOST_CHECK(chunk[0] == trainingData[4]);
      BOOST_CHECK(chunk[1] == trainingData[5]);
      BOOST_CHECK(chunk[2] == trainingData[6]);
      BOOST_CHECK(chunk[3] == trainingData[7]);

      BOOST_CHECK(partial_final_graph[0] == final_graph[4]);
      BOOST_CHECK(partial_final_graph[1] == final_graph[5]);
      BOOST_CHECK(partial_final_graph[2] == final_graph[6]);
      BOOST_CHECK(partial_final_graph[3] == final_graph[7]);

      BOOST_CHECK(partial_final_graph_distances[0] == final_graph_distances[4]);
      BOOST_CHECK(partial_final_graph_distances[1] == final_graph_distances[5]);
      BOOST_CHECK(partial_final_graph_distances[2] == final_graph_distances[6]);
      BOOST_CHECK(partial_final_graph_distances[3] == final_graph_distances[7]);

      BOOST_CHECK(partial_indices_map[0] == 2);
      BOOST_CHECK(partial_indices_map[1] == 3);
    } else {
      BOOST_FAIL("Should never get here");
    }
  }
}

BOOST_AUTO_TEST_CASE(merge_chunk) {
  dim = 2;
  k = 2;
  dataset_count = 4;
  size_t chunk_range = 2;

  sgpp::base::DataMatrix trainingData(dataset_count, dim);
  trainingData[0] = 1.0;
  trainingData[1] = 2.0;
  trainingData[2] = 3.0;
  trainingData[3] = 4.0;
  trainingData[4] = 5.0;
  trainingData[5] = 6.0;
  trainingData[6] = 7.0;
  trainingData[7] = 8.0;
  sgpp::datadriven::OperationNearestNeighborSampled knn_op(true);

  std::vector<int64_t> final_graph{
      1, 2, //
      2, 3, //
      3, 4, //
      4, 5  //
  };
  std::vector<int64_t> final_graph_copy(final_graph);
  std::vector<double> final_graph_distances{
      0.1, 0.2, //
      0.3, 0.4, //
      0.5, 0.6, //
      0.7, 0.8  //
  };
  std::vector<double> final_graph_distances_copy(final_graph_distances);

  std::vector<int64_t> indices_map{0, 1, 2, 3};

  for (size_t chunk_first_index = 0; chunk_first_index < dataset_count;
       chunk_first_index += chunk_range) {
    auto [chunk, partial_final_graph, partial_final_graph_distances,
          partial_indices_map] =
        knn_op.extract_chunk(dim, k, chunk_first_index, chunk_range,
                             trainingData, final_graph, final_graph_distances,
                             indices_map);
    for (size_t i = 0; i < chunk_range; i += 1) {
      for (size_t j = 0; j < k; j += 1) {
        partial_final_graph[i * k + j] *= 10;
        partial_final_graph_distances[i * k + j] /= 10.0;
      }
    }

    // std::cout << "partial_final_graph:" << std::endl;
    // for (size_t i = 0; i < chunk_range; i += 1) {
    //   for (size_t j = 0; j < k; j += 1) {
    //     if (j > 0)
    //       std::cout << ", ";
    //     std::cout << partial_final_graph[i * k + j];
    //   }
    //   std::cout << std::endl;
    // }
    // std::cout << "partial_final_graph_distances:" << std::endl;
    // for (size_t i = 0; i < chunk_range; i += 1) {
    //   for (size_t j = 0; j < k; j += 1) {
    //     if (j > 0)
    //       std::cout << ", ";
    //     std::cout << partial_final_graph_distances[i * k + j];
    //   }
    //   std::cout << std::endl;
    // }

    knn_op.merge_chunk(k, chunk_first_index, chunk_range, final_graph,
                       final_graph_distances, partial_final_graph,
                       partial_final_graph_distances);

    // std::cout << "final_graph after merge:" << std::endl;
    // for (size_t i = 0; i < dataset_count; i += 1) {
    //   for (size_t j = 0; j < k; j += 1) {
    //     if (j > 0)
    //       std::cout << ", ";
    //     std::cout << final_graph[i * k + j];
    //   }
    //   std::cout << std::endl;
    // }
    // std::cout << "final_graph_distances after merge:" << std::endl;
    // for (size_t i = 0; i < dataset_count; i += 1) {
    //   for (size_t j = 0; j < k; j += 1) {
    //     if (j > 0)
    //       std::cout << ", ";
    //     std::cout << final_graph_distances[i * k + j];
    //   }
    //   std::cout << std::endl;
    // }

    for (size_t i = 0; i < chunk_range; i += 1) {
      std::sort(final_graph.begin() + (chunk_first_index + i * k),
                final_graph.begin() + (chunk_first_index + i * k + k));
      std::sort(final_graph_distances.begin() + (chunk_first_index + i * k),
                final_graph_distances.begin() +
                    (chunk_first_index + i * k + k));
    }

    if (chunk_first_index == 0) {
      BOOST_CHECK(final_graph[0] == 10);
      BOOST_CHECK(final_graph[1] == 20);
      BOOST_CHECK(final_graph[2] == 20);
      BOOST_CHECK(final_graph[3] == 30);

      BOOST_CHECK_CLOSE(final_graph_distances[0], 0.01, 1E-5);
      BOOST_CHECK_CLOSE(final_graph_distances[1], 0.02, 1E-5);
      BOOST_CHECK_CLOSE(final_graph_distances[2], 0.03, 1E-5);
      BOOST_CHECK_CLOSE(final_graph_distances[3], 0.04, 1E-5);
    } else if (chunk_first_index == 2) {
      BOOST_CHECK(final_graph[4] == 30);
      BOOST_CHECK(final_graph[5] == 40);
      BOOST_CHECK(final_graph[6] == 40);
      BOOST_CHECK(final_graph[7] == 50);

      BOOST_CHECK_CLOSE(final_graph_distances[4], 0.05, 1E-5);
      BOOST_CHECK_CLOSE(final_graph_distances[5], 0.06, 1E-5);
      BOOST_CHECK_CLOSE(final_graph_distances[6], 0.07, 1E-5);
      BOOST_CHECK_CLOSE(final_graph_distances[7], 0.08, 1E-5);
    } else {
      BOOST_FAIL("Should never get here");
    }
  }
}

BOOST_AUTO_TEST_CASE(merge_knn_second_round_two_ranges) {

  dim = 2;
  k = 2;
  dataset_count = 4;
  size_t chunk_range = 2;

  sgpp::base::DataMatrix trainingData(dataset_count, dim);
  // first data tuple
  trainingData[0] = 1.0;
  trainingData[1] = 1.0;
  // second data tuple
  trainingData[2] = 1.0;
  trainingData[3] = 1.0;
  // third data tuple
  trainingData[4] = 0.0;
  trainingData[5] = 0.0;
  // fourth data tuple
  trainingData[6] = 0.0;
  trainingData[7] = 0.0;

  sgpp::datadriven::OperationNearestNeighborSampled knn_op(true);

  std::vector<int64_t> final_graph{
      2, 3, //
      2, 3, //
      0, 1, //
      0, 1  //
  };
  constexpr double init = sqrt(2);
  std::vector<double> final_graph_distances{
      init, init, //
      init, init, //
      init, init, //
      init, init  //
  };

  std::vector<int64_t> partial_graphs[2] = {{
                                                0, 1, //
                                                0, 1, //
                                            },
                                            {
                                                0, 1, // act: 2, 3
                                                0, 1, // act: 2, 3
                                            }};

  std::vector<int64_t> indices_map{0, 1, 2, 3};

  size_t round = 0;
  for (size_t chunk_first_index = 0; chunk_first_index < dataset_count;
       chunk_first_index += chunk_range) {
    // std::cout << "------------- begin round --------------" << std::endl;
    // std::cout << "chunk_first_index: " << chunk_first_index << std::endl;
    std::vector<int64_t> &partial_graph{partial_graphs[round]};

    auto [chunk, partial_final_graph, partial_final_graph_distances,
          partial_indices_map] =
        knn_op.extract_chunk(dim, k, chunk_first_index, chunk_range,
                             trainingData, final_graph, final_graph_distances,
                             indices_map);

    // std::cout << "partial_final_graph:" << std::endl;
    // for (size_t i = 0; i < chunk_range; i += 1) {
    //   for (size_t j = 0; j < k; j += 1) {
    //     if (j > 0) {
    //       std::cout << ", ";
    //     }
    //     std::cout << partial_final_graph[i * k + j];
    //   }
    //   std::cout << std::endl;
    // }
    // std::cout << "partial_final_graph_distances:" << std::endl;
    // for (size_t i = 0; i < chunk_range; i += 1) {
    //   for (size_t j = 0; j < k; j += 1) {
    //     if (j > 0) {
    //       std::cout << ", ";
    //     }
    //     std::cout << partial_final_graph_distances[i * k + j];
    //   }
    //   std::cout << std::endl;
    // }
    // std::cout << "partial_graph:" << std::endl;
    // for (size_t i = 0; i < chunk_range; i += 1) {
    //   for (size_t j = 0; j < k; j += 1) {
    //     if (j > 0) {
    //       std::cout << ", ";
    //     }
    //     std::cout << partial_graph[i * k + j];
    //   }
    //   std::cout << std::endl;
    // }

    // std::cout << "-------------- merge_knn ---------------" << std::endl;

    knn_op.merge_knn(dim, k, chunk, chunk_range, partial_final_graph,
                     partial_final_graph_distances, partial_graph,
                     partial_indices_map);
    // std::cout << "partial_final_graph:" << std::endl;
    // for (size_t i = 0; i < chunk_range; i += 1) {
    //   for (size_t j = 0; j < k; j += 1) {
    //     if (j > 0) {
    //       std::cout << ", ";
    //     }
    //     std::cout << partial_final_graph[i * k + j];
    //   }
    //   std::cout << std::endl;
    // }
    // std::cout << "partial_final_graph_distances:" << std::endl;
    // for (size_t i = 0; i < chunk_range; i += 1) {
    //   for (size_t j = 0; j < k; j += 1) {
    //     if (j > 0) {
    //       std::cout << ", ";
    //     }
    //     std::cout << partial_final_graph_distances[i * k + j];
    //   }
    //   std::cout << std::endl;
    // }
    // std::cout << "------------- end output round --------------" <<
    // std::endl;

    if (round == 0) {
      BOOST_CHECK(partial_final_graph[0] == 0);
      BOOST_CHECK(partial_final_graph[1] == 1);
      BOOST_CHECK(partial_final_graph[2] == 0);
      BOOST_CHECK(partial_final_graph[3] == 1);
    } else if (round == 1) {
      BOOST_CHECK(partial_final_graph[0] == 2);
      BOOST_CHECK(partial_final_graph[1] == 3);
      BOOST_CHECK(partial_final_graph[2] == 2);
      BOOST_CHECK(partial_final_graph[3] == 3);
    } else {
      BOOST_FAIL("Should never get here");
    }

    if (round == 0) {
      BOOST_CHECK(partial_final_graph_distances[0] == 0.0);
      BOOST_CHECK(partial_final_graph_distances[1] == 0.0);
      BOOST_CHECK(partial_final_graph_distances[2] == 0.0);
      BOOST_CHECK(partial_final_graph_distances[3] == 0.0);
    } else if (round == 1) {
      BOOST_CHECK(partial_final_graph_distances[0] == 0.0);
      BOOST_CHECK(partial_final_graph_distances[1] == 0.0);
      BOOST_CHECK(partial_final_graph_distances[2] == 0.0);
      BOOST_CHECK(partial_final_graph_distances[3] == 0.0);
    } else {
      BOOST_FAIL("Should never get here");
    }

    round += 1;
  }
}

BOOST_AUTO_TEST_CASE(merge_chunk_randomize) {
  dim = 2;
  k = 2;
  dataset_count = 4;
  size_t chunk_range = 2;

  sgpp::base::DataMatrix trainingData(dataset_count, dim);
  trainingData[0] = 1.0;
  trainingData[1] = 2.0;
  trainingData[2] = 3.0;
  trainingData[3] = 4.0;
  trainingData[4] = 5.0;
  trainingData[5] = 6.0;
  trainingData[6] = 7.0;
  trainingData[7] = 8.0;
  sgpp::datadriven::OperationNearestNeighborSampled knn_op(true);

  std::vector<int64_t> final_graph{
      1, 2, //
      2, 3, //
      3, 4, //
      4, 5  //
  };
  // std::vector<int64_t> final_graph_copy(final_graph);
  std::vector<double> final_graph_distances{
      0.1, 0.2, //
      0.3, 0.4, //
      0.5, 0.6, //
      0.7, 0.8  //
  };
  // std::vector<double> final_graph_distances_copy(final_graph_distances);

  std::vector<int64_t> indices_map{0, 1, 2, 3};

  knn_op.randomize(dim, trainingData, k, final_graph, final_graph_distances,
                   indices_map);

  for (size_t chunk_first_index = 0; chunk_first_index < dataset_count;
       chunk_first_index += chunk_range) {
    auto [chunk, partial_final_graph, partial_final_graph_distances,
          partial_indices_map] =
        knn_op.extract_chunk(dim, k, chunk_first_index, chunk_range,
                             trainingData, final_graph, final_graph_distances,
                             indices_map);
    for (size_t i = 0; i < chunk_range; i += 1) {
      for (size_t j = 0; j < k; j += 1) {
        partial_final_graph[i * k + j] *= 10;
        partial_final_graph_distances[i * k + j] /= 10.0;
      }
    }

    knn_op.merge_chunk(k, chunk_first_index, chunk_range, final_graph,
                       final_graph_distances, partial_final_graph,
                       partial_final_graph_distances);
  }

  knn_op.undo_randomize(dim, trainingData, k, final_graph,
                        final_graph_distances, indices_map);

  for (size_t i = 0; i < dataset_count; i += 1) {
    std::sort(final_graph.begin() + (i * k), final_graph.begin() + (i * k + k));
  }
  for (size_t i = 0; i < dataset_count; i += 1) {
    std::sort(final_graph_distances.begin() + (i * k),
              final_graph_distances.begin() + (i * k + k));
  }

  BOOST_CHECK(final_graph[0] == 10);
  BOOST_CHECK(final_graph[1] == 20);
  BOOST_CHECK(final_graph[2] == 20);
  BOOST_CHECK(final_graph[3] == 30);
  BOOST_CHECK(final_graph[4] == 30);
  BOOST_CHECK(final_graph[5] == 40);
  BOOST_CHECK(final_graph[6] == 40);
  BOOST_CHECK(final_graph[7] == 50);

  BOOST_CHECK_CLOSE(final_graph_distances[0], 0.01, 1E-5);
  BOOST_CHECK_CLOSE(final_graph_distances[1], 0.02, 1E-5);
  BOOST_CHECK_CLOSE(final_graph_distances[2], 0.03, 1E-5);
  BOOST_CHECK_CLOSE(final_graph_distances[3], 0.04, 1E-5);
  BOOST_CHECK_CLOSE(final_graph_distances[4], 0.05, 1E-5);
  BOOST_CHECK_CLOSE(final_graph_distances[5], 0.06, 1E-5);
  BOOST_CHECK_CLOSE(final_graph_distances[6], 0.07, 1E-5);
  BOOST_CHECK_CLOSE(final_graph_distances[7], 0.08, 1E-5);
}

BOOST_AUTO_TEST_CASE(merge_knn_minus_one_randomize) {

  dim = 2;
  k = 2;
  dataset_count = 4;
  size_t chunk_range = 4;
  sgpp::base::DataMatrix chunk(dataset_count, dim);
  for (size_t i = 0; i < dataset_count; i += 1) {
    for (size_t d = 0; d < dim; d += 1) {
      chunk[i * dim + d] = 0.5;
    }
  }

  std::vector<int64_t> partial_final_graph{
      -1, -1, //
      -1, -1, //
      -1, -1, //
      -1, -1, //

  };
  std::vector<double> partial_final_graph_distances{
      4, 4, //
      4, 4, //
      4, 4, //
      4, 4, //

  };
  std::vector<int64_t> partial_graph{
      0, 1, //
      0, 1, //
      0, 1, //
      0, 1, //
  };

  std::vector<int64_t> partial_indices_map{3, 0, 2, 1};

  sgpp::datadriven::OperationNearestNeighborSampled knn_op(true);

  for (size_t chunk_first_index = 0; chunk_first_index < dataset_count;
       chunk_first_index += chunk_range) {

    knn_op.merge_knn(dim, k, chunk, chunk_range, partial_final_graph,
                     partial_final_graph_distances, partial_graph,
                     partial_indices_map);
    // std::cout << "partial_final_graph after merge_knn:" << std::endl;
    // for (size_t i = 0; i < chunk_range; i += 1) {
    //   for (size_t j = 0; j < k; j += 1) {
    //     if (j > 0)
    //       std::cout << ", ";
    //     std::cout << partial_final_graph[i * k + j];
    //   }
    //   std::cout << std::endl;
    // }
    // std::cout << "partial_final_graph_distances after merge_knn:" <<
    // std::endl; for (size_t i = 0; i < chunk_range; i += 1) {
    //   for (size_t j = 0; j < k; j += 1) {
    //     if (j > 0)
    //       std::cout << ", ";
    //     std::cout << partial_final_graph_distances[i * k + j];
    //   }
    //   std::cout << std::endl;
    // }
    for (size_t i = 0; i < chunk_range; i += 1) {
      for (size_t j = 0; j < k; j += 1) {
        if (j == 0) {
          BOOST_CHECK(partial_final_graph[i * k + j] == 3);
          BOOST_CHECK(partial_final_graph_distances[i * k + j] == 0.0);
        } else {
          BOOST_CHECK(partial_final_graph[i * k + j] == 0);
          BOOST_CHECK(partial_final_graph_distances[i * k + j] == 0.0);
        }
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(merge_knn_two_round_randomize) {

  dim = 2;
  k = 2;
  dataset_count = 4;
  size_t chunk_range = 2;
  std::array<sgpp::base::DataMatrix, 2> chunks;
  chunks[0] = sgpp::base::DataMatrix(dataset_count, dim);
  chunks[0][0] = 0.5;
  chunks[0][1] = 0.5;
  chunks[0][2] = 0.5;
  chunks[0][3] = 0.5;

  chunks[1] = sgpp::base::DataMatrix(dataset_count, dim);
  chunks[1][0] = 0.5;
  chunks[1][1] = 0.5;
  chunks[1][2] = 0.5;
  chunks[1][3] = 0.5;

  std::array<std::vector<int64_t>, 2> partial_final_graphs{{
      {-1, -1,  //
       -1, -1}, //
      {-1, -1,  //
       -1, -1}  //
  }};
  std::array<std::vector<double>, 2> partial_final_graph_distances{{
      {4, 4,  //
       4, 4}, //
      {4, 4,  //
       4, 4}  //
  }};
  std::array<std::vector<int64_t>, 2> partial_graphs{{
      {0, 1,  //
       0, 1}, //
      {0, 1,  //
       0, 1}  //
  }};

  std::array<std::vector<int64_t>, 2> partial_indices_maps{{
      {3, 0}, //
      {2, 1}  //
  }};

  sgpp::datadriven::OperationNearestNeighborSampled knn_op(true);

  size_t round = 0;
  for (size_t chunk_first_index = 0; chunk_first_index < dataset_count;
       chunk_first_index += chunk_range) {

    // std::cout << "partial_graphs before merge_knn:" << std::endl;
    // for (size_t i = 0; i < chunk_range; i += 1) {
    //   for (size_t j = 0; j < k; j += 1) {
    //     if (j > 0)
    //       std::cout << ", ";
    //     std::cout << partial_graphs[round][i * k + j];
    //   }
    //   std::cout << std::endl;
    // }
    // std::cout << "partial_final_graph before merge_knn:" << std::endl;
    // for (size_t i = 0; i < chunk_range; i += 1) {
    //   for (size_t j = 0; j < k; j += 1) {
    //     if (j > 0)
    //       std::cout << ", ";
    //     std::cout << partial_final_graphs[round][i * k + j];
    //   }
    //   std::cout << std::endl;
    // }
    // std::cout << "partial_final_graph_distances before merge_knn:" <<
    // std::endl; for (size_t i = 0; i < chunk_range; i += 1) {
    //   for (size_t j = 0; j < k; j += 1) {
    //     if (j > 0)
    //       std::cout << ", ";
    //     std::cout << partial_final_graph_distances[round][i * k + j];
    //   }
    //   std::cout << std::endl;
    // }

    knn_op.merge_knn(dim, k, chunks[round], chunk_range,
                     partial_final_graphs[round],
                     partial_final_graph_distances[round],
                     partial_graphs[round], partial_indices_maps[round]);
    // std::cout << "partial_final_graph after merge_knn:" << std::endl;
    // for (size_t i = 0; i < chunk_range; i += 1) {
    //   for (size_t j = 0; j < k; j += 1) {
    //     if (j > 0)
    //       std::cout << ", ";
    //     std::cout << partial_final_graphs[round][i * k + j];
    //   }
    //   std::cout << std::endl;
    // }
    // std::cout << "partial_final_graph_distances after merge_knn:" <<
    // std::endl; for (size_t i = 0; i < chunk_range; i += 1) {
    //   for (size_t j = 0; j < k; j += 1) {
    //     if (j > 0)
    //       std::cout << ", ";
    //     std::cout << partial_final_graph_distances[round][i * k + j];
    //   }
    //   std::cout << std::endl;
    // }
    for (size_t i = 0; i < chunk_range; i += 1) {
      for (size_t j = 0; j < k; j += 1) {
        if (j == 0) {
          if (chunk_first_index == 2) {
            BOOST_CHECK(partial_final_graphs[round][i * k + j] == 2);
          } else {
            BOOST_CHECK(partial_final_graphs[round][i * k + j] == 3);
          }
          BOOST_CHECK(partial_final_graph_distances[round][i * k + j] == 0.0);
        } else {
          if (chunk_first_index == 2) {
            BOOST_CHECK(partial_final_graphs[round][i * k + j] == 1);
          } else {
            BOOST_CHECK(partial_final_graphs[round][i * k + j] == 0);
          }
          BOOST_CHECK(partial_final_graph_distances[round][i * k + j] == 0.0);
        }
      }
    }

    round += 1;
  }
  // for (size_t i = 0; i < dataset_count; i += 1) {
  // }
}

BOOST_AUTO_TEST_SUITE_END()

#endif
