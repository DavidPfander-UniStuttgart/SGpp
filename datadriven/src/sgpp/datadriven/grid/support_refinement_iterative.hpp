#pragma once

#include <chrono>
#include <cinttypes>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#if USE_OCL == 1
#include "sgpp/base/opencl/manager/managed_buffer.hpp"
#include "sgpp/base/opencl/manager/manager.hpp"
#endif

namespace sgpp::datadriven {

class support_refinement_iterative {
 private:
  int64_t dim;
  int64_t max_level;
  int64_t min_support;
  std::vector<double> data;

  int64_t entries;
  std::vector<int64_t> levels;
  std::vector<int64_t> indices;
  int64_t num_visited;

  std::vector<int64_t> schedule_level;
  std::vector<int64_t> schedule_index;
  std::vector<bool> schedule_support;

  std::vector<int64_t> schedule_next_level;
  std::vector<int64_t> schedule_next_index;

  enum class direction { left, right };

#if USE_OCL == 1
  bool use_ocl = false;
  std::string ocl_config_file_name;
  const int local_cache = 16;
  const int block_size = 128;
  std::unique_ptr<opencl::manager_t> manager;
  std::vector<cl_kernel> kernels_verify_support;
  std::vector<std::unique_ptr<opencl::managed_buffer<double>>> data_devices;

#endif

  inline void child_d_dir(std::vector<int64_t> &l, std::vector<int64_t> &i,
                          std::vector<int64_t> &child_l, std::vector<int64_t> &child_i, int64_t d,
                          direction dir) {
    child_l = l;
    child_i = i;
    child_l[d] += 1;
    if (dir == direction::left) {
      child_i[d] = 2 * child_i[d] - 1;
    } else if (dir == direction::right) {
      child_i[d] = 2 * child_i[d] + 1;
    } else {
      throw;  // never
    }
  }

  inline void neighbor_pos_dir(std::vector<int64_t> &l, std::vector<int64_t> &i,
                               std::vector<double> &neighbor, direction dir) {
    if (dir == direction::left) {
      for (int64_t d = 0; d < dim; d += 1) {
        neighbor[d] = std::pow(2.0, static_cast<double>(-l[d])) * static_cast<double>(i[d] - 1ll);
      }
    } else if (dir == direction::right) {
      for (int64_t d = 0; d < dim; d += 1) {
        neighbor[d] = std::pow(2.0, static_cast<double>(-l[d])) * static_cast<double>(i[d] + 1ll);
      }
    } else {
      throw;  // never
    }
  }

  void neighbor_pos_dir_blocked(std::vector<int64_t> &schedule_level,
                                std::vector<int64_t> &schedule_index,
                                std::vector<double> &neighbors, direction dir);

  void verify_support_blocked();

#if USE_OCL == 1
  void verify_support_blocked_OCL();
#endif

  void refine_impl();

 public:
  support_refinement_iterative(int64_t dim, int64_t max_level, int64_t min_support,
                               const std::vector<double> &data);
  ~support_refinement_iterative();
  void refine();
  std::vector<int64_t> &get_levels();
  std::vector<int64_t> &get_indices();
  void print_level(std::vector<int64_t> l);
  void print_index(std::vector<int64_t> i);
  void print_pos(std::vector<double> pos);
  static void write_grid_positions(int64_t dim, const std::string &grid_file_name,
                                   const std::string &dataset_file_name, std::vector<int64_t> &ls,
                                   std::vector<int64_t> &is);

#if USE_OCL == 1
  void enable_OCL(const std::string &ocl_config_file_name);
#endif

};  // namespace datadriven

}  // namespace sgpp::datadriven
