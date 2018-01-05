#include "parameters.hpp"

#include <cstddef>
#include <vector>

#include <Vc/Vc>
using Vc::double_v;

#include <opttmp/vectorization/register_tiling.hpp>
using namespace opttmp::vectorization;

#include <omp.h>

extern "C" void streaming_mult_kernel(size_t dims, std::vector<double> &dataset_SoA,
                                      size_t dataset_size, std::vector<double> &level_list,
                                      std::vector<double> &index_list, std::vector<double> &alpha,
                                      std::vector<double> &result_padded) {
// omp_set_num_threads(KERNEL_OMP_THREADS);
// schedule(static) num_threads(KERNEL_OMP_THREADS)
#pragma omp parallel for
  for (size_t i = 0; i < dataset_size; i += DATA_BLOCKING * double_v::size()) {
    register_array<double_v, DATA_BLOCKING> result_temps_arr(0.0);  // broadcast

    for (size_t j = 0; j < alpha.size(); j++) {
      register_array<double_v, DATA_BLOCKING> evalNds_arr(alpha[j]);  // broadcast

      for (size_t d = 0; d < dims; d++) {
        // non-SoA is faster (2 streams, instead of 2d streams)
        // 2^l * x - i (level_list_SoA stores 2^l, not l)
        // double_v level_dim = level_list[j * dims + d];  // broadcasts
        // double_v index_dim = index_list[j * dims + d];

        register_array<double_v, DATA_BLOCKING> data_dims_arr(&dataset_SoA[d * dataset_size + i],
                                                              Vc::flags::element_aligned);

        register_array<double_v, DATA_BLOCKING> temps_arr;
        // converted to FMA through expression templates
        // (also mixes array and non-array type vector variables)
        // temps_arr = (data_dims_arr * level_dim) - index_dim;  // 2 FLOP
        temps_arr = (data_dims_arr * double_v(level_list[j * dims + d])) -
                    double_v(index_list[j * dims + d]);  // 2 FLOP

        register_array<double_v, DATA_BLOCKING> eval1ds_arr;
        eval1ds_arr = opttmp::vectorization::max(
            double_v(1.0) - opttmp::vectorization::abs(temps_arr), double_v(0.0));  // 3 FLOP

        evalNds_arr *= eval1ds_arr;  // 1 FLOP
      }
      result_temps_arr += evalNds_arr;
    }

    result_temps_arr.memstore(&result_padded[i], Vc::flags::element_aligned);
  }

  //   constexpr size_t additional_data_blocking = 2 * DATA_BLOCKING * double_v::size();
  //   constexpr size_t additional_grid_blocking = 10;

  // #pragma omp parallel for
  //   for (size_t ii = 0; ii < dataset_size; ii += additional_data_blocking) {
  //     for (size_t jj = 0; jj < alpha.size(); jj += additional_grid_blocking) {
  //       for (size_t i = ii; i < std::min(ii + additional_data_blocking, dataset_size);
  //            i += DATA_BLOCKING * double_v::size()) {
  //         for (size_t j = jj; j < std::min(jj + additional_grid_blocking, alpha.size()); j++) {
  //           register_array<double_v, DATA_BLOCKING> evalNds_arr(alpha[j]);  // broadcast

  //           for (size_t d = 0; d < dims; d++) {
  //             // non-SoA is faster (2 streams, instead of 2d streams)
  //             // 2^l * x - i (level_list_SoA stores 2^l, not l)
  //             // double_v level_dim = level_list[j * dims + d];  // broadcasts
  //             // double_v index_dim = index_list[j * dims + d];

  //             register_array<double_v, DATA_BLOCKING> data_dims_arr(
  //                 &dataset_SoA[d * dataset_size + i], Vc::flags::element_aligned);

  //             register_array<double_v, DATA_BLOCKING> temps_arr;
  //             // converted to FMA through expression templates
  //             // (also mixes array and non-array type vector variables)
  //             // temps_arr = (data_dims_arr * level_dim) - index_dim;  // 2 FLOP
  //             temps_arr = (data_dims_arr * double_v(level_list[j * dims + d])) -
  //                         double_v(index_list[j * dims + d]);  // 2 FLOP

  //             register_array<double_v, DATA_BLOCKING> eval1ds_arr;
  //             eval1ds_arr = opttmp::vectorization::max(
  //                 double_v(1.0) - opttmp::vectorization::abs(temps_arr), double_v(0.0));  // 3
  //                 FLOP

  //             evalNds_arr *= eval1ds_arr;  // 1 FLOP
  //           }
  //           register_array<double_v, DATA_BLOCKING> cur_result_arr(
  //               &result_padded[i],
  //               Vc::flags::element_aligned);  // broadcast
  //           cur_result_arr += evalNds_arr;
  //           cur_result_arr.memstore(&result_padded[i], Vc::flags::element_aligned);
  //         }
  //       }
  //     }
  //   }
}
