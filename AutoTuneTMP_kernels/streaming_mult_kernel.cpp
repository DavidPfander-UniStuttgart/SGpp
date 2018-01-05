#include "parameters.hpp"

#include <cstddef>
#include <vector>

#include <Vc/Vc>
using Vc::double_v;

#include <opttmp/vectorization/register_tiling.hpp>
using namespace opttmp::vectorization;

using reg_array = register_array<double_v, DATA_BLOCKING>;

#include <omp.h>

#include "sgpp/base/operation/hash/OperationMultipleEval.hpp"

// sgpp::base::Grid& grid, base::DataMatrix& dataset, sgpp::base::DataVector& alpha,
// sgpp::base::DataVector& result

extern "C" void streaming_mult_kernel(size_t dims, sgpp::base::Grid& grid,
                                      sgpp::base::DataMatrix& dataset,
                                      sgpp::base::DataVector& alpha, sgpp::base::DataVector& result)

{
  ////////////////////////////////////////
  // prepare data structures
  ////////////////////////////////////////

  // grid as SoA
  sgpp::base::GridStorage& storage = grid.getStorage();
  // grid as non-SoA for testing
  std::vector<double> level_list;
  std::vector<double> index_list;
  level_list.resize(storage.getSize() * dims);
  index_list.resize(storage.getSize() * dims);
  for (size_t i = 0; i < storage.getSize(); i++) {
    sgpp::base::HashGridPoint& point = storage[i];
    for (size_t d = 0; d < dims; d++) {
      sgpp::base::GridPoint::level_type level;
      sgpp::base::GridPoint::index_type index;
      point.get(d, level, index);
      level_list[i * dims + d] = static_cast<double>(1 << level);
      index_list[i * dims + d] = static_cast<double>(index);
    }
  }

  // data as SoA
  size_t dataset_size = dataset.getNrows();
  dataset_size += dataset_size % (DATA_BLOCKING * double_v::size()) == 0
                      ? 0
                      : (DATA_BLOCKING * double_v::size()) -
                            (dataset_size % (DATA_BLOCKING * double_v::size()));
  std::vector<double> dataset_SoA;
  dataset_SoA.resize(dataset_size * dims);
  for (size_t i = 0; i < dataset.getNrows(); i++) {
    for (size_t d = 0; d < dims; d++) {
      dataset_SoA[d * dataset_size + i] = dataset[i * dims + d];
    }
  }
  // setup padding
  for (size_t i = dataset.getNrows(); i < dataset_size; i++) {
    for (size_t d = 0; d < dims; d++) {
      dataset_SoA[d * dataset_size + i] = dataset[(dataset.getNrows() - 1) * dims + d];
    }
  }

  std::vector<double> result_padded(dataset_size);
  std::fill(result_padded.begin(), result_padded.begin(), 0.0);

////////////////////////////////////////
// end prepare data structures
////////////////////////////////////////

#pragma omp parallel for num_threads(KERNEL_OMP_THREADS)
  for (size_t i = 0; i < dataset_size; i += DATA_BLOCKING * double_v::size()) {
    reg_array result_temps_arr(0.0);  // broadcast

    for (size_t j = 0; j < alpha.size(); j++) {
      reg_array evalNds_arr(alpha[j]);  // broadcast

      for (size_t d = 0; d < dims; d++) {
        // non-SoA is faster (2 streams, instead of 2d streams)
        // 2^l * x - i (level_list_SoA stores 2^l, not l)
        // double_v level_dim = level_list[j * dims + d];  // broadcasts
        // double_v index_dim = index_list[j * dims + d];

        reg_array data_dims_arr(&dataset_SoA[d * dataset_size + i], Vc::flags::element_aligned);

        reg_array temps_arr;
        // converted to FMA through expression templates
        // (also mixes array and non-array type vector variables)
        // temps_arr = (data_dims_arr * level_dim) - index_dim;  // 2 FLOP
        temps_arr = (data_dims_arr * double_v(level_list[j * dims + d])) -
                    double_v(index_list[j * dims + d]);  // 2 FLOP

        reg_array eval1ds_arr;
        eval1ds_arr = opttmp::vectorization::max(
            double_v(1.0) - opttmp::vectorization::abs(temps_arr), double_v(0.0));  // 3 FLOP

        evalNds_arr *= eval1ds_arr;  // 1 FLOP
      }
      result_temps_arr += evalNds_arr;
    }

    result_temps_arr.memstore(&result_padded[i], Vc::flags::element_aligned);
  }

  for (size_t i = 0; i < result.size(); i++) {
    result[i] = result_padded[i];
  }
}

// extern "C" void streaming_mult_kernel(size_t dims, std::vector<double> &dataset_SoA,
//                                       size_t dataset_size, std::vector<double> &level_list,
//                                       std::vector<double> &index_list, std::vector<double>
//                                       &alpha,
//                                       std::vector<double> &result_padded) {
// #pragma omp parallel for num_threads(KERNEL_OMP_THREADS)
//   for (size_t i = 0; i < dataset_size; i += DATA_BLOCKING * double_v::size()) {
//     reg_array result_temps_arr(0.0);  // broadcast

//     for (size_t j = 0; j < alpha.size(); j++) {
//       reg_array evalNds_arr(alpha[j]);  // broadcast

//       for (size_t d = 0; d < dims; d++) {
//         // non-SoA is faster (2 streams, instead of 2d streams)
//         // 2^l * x - i (level_list_SoA stores 2^l, not l)
//         // double_v level_dim = level_list[j * dims + d];  // broadcasts
//         // double_v index_dim = index_list[j * dims + d];

//         reg_array data_dims_arr(&dataset_SoA[d * dataset_size + i], Vc::flags::element_aligned);

//         reg_array temps_arr;
//         // converted to FMA through expression templates
//         // (also mixes array and non-array type vector variables)
//         // temps_arr = (data_dims_arr * level_dim) - index_dim;  // 2 FLOP
//         temps_arr = (data_dims_arr * double_v(level_list[j * dims + d])) -
//                     double_v(index_list[j * dims + d]);  // 2 FLOP

//         reg_array eval1ds_arr;
//         eval1ds_arr = opttmp::vectorization::max(
//             double_v(1.0) - opttmp::vectorization::abs(temps_arr), double_v(0.0));  // 3 FLOP

//         evalNds_arr *= eval1ds_arr;  // 1 FLOP
//       }
//       result_temps_arr += evalNds_arr;
//     }

//     result_temps_arr.memstore(&result_padded[i], Vc::flags::element_aligned);
//   }

//   //   constexpr size_t additional_data_blocking = 2 * DATA_BLOCKING * double_v::size();
//   //   constexpr size_t additional_grid_blocking = 10;

//   // #pragma omp parallel for
//   //   for (size_t ii = 0; ii < dataset_size; ii += additional_data_blocking) {
//   //     for (size_t jj = 0; jj < alpha.size(); jj += additional_grid_blocking) {
//   //       for (size_t i = ii; i < std::min(ii + additional_data_blocking, dataset_size);
//   //            i += DATA_BLOCKING * double_v::size()) {
//   //         for (size_t j = jj; j < std::min(jj + additional_grid_blocking, alpha.size()); j++)
//   {
//   //           reg_array evalNds_arr(alpha[j]);  // broadcast

//   //           for (size_t d = 0; d < dims; d++) {
//   //             // non-SoA is faster (2 streams, instead of 2d streams)
//   //             // 2^l * x - i (level_list_SoA stores 2^l, not l)
//   //             // double_v level_dim = level_list[j * dims + d];  // broadcasts
//   //             // double_v index_dim = index_list[j * dims + d];

//   //             reg_array data_dims_arr(
//   //                 &dataset_SoA[d * dataset_size + i], Vc::flags::element_aligned);

//   //             reg_array temps_arr;
//   //             // converted to FMA through expression templates
//   //             // (also mixes array and non-array type vector variables)
//   //             // temps_arr = (data_dims_arr * level_dim) - index_dim;  // 2 FLOP
//   //             temps_arr = (data_dims_arr * double_v(level_list[j * dims + d])) -
//   //                         double_v(index_list[j * dims + d]);  // 2 FLOP

//   //             reg_array eval1ds_arr;
//   //             eval1ds_arr = opttmp::vectorization::max(
//   //                 double_v(1.0) - opttmp::vectorization::abs(temps_arr), double_v(0.0));  // 3
//   //                 FLOP

//   //             evalNds_arr *= eval1ds_arr;  // 1 FLOP
//   //           }
//   //           reg_array cur_result_arr(
//   //               &result_padded[i],
//   //               Vc::flags::element_aligned);  // broadcast
//   //           cur_result_arr += evalNds_arr;
//   //           cur_result_arr.memstore(&result_padded[i], Vc::flags::element_aligned);
//   //         }
//   //       }
//   //     }
//   //   }
// }
