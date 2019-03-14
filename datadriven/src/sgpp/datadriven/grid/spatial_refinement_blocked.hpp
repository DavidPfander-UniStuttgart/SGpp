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

class spatial_refinement_blocked {
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
  const int block_size = 256;
  // std::string configuration_file(ocl_config_file_name);
  std::unique_ptr<opencl::manager_t> manager;
  opencl::device_t device;
  cl_kernel kernel_verify_support = nullptr;
  std::unique_ptr<opencl::managed_buffer<double>> data_device;

#endif

  inline void child_d_dir(std::vector<int64_t> &l, std::vector<int64_t> &i,
                          std::vector<int64_t> &child_l,
                          std::vector<int64_t> &child_i, int64_t d,
                          direction dir) {
    child_l = l;
    child_i = i;
    child_l[d] += 1;
    if (dir == direction::left) {
      child_i[d] = 2 * child_i[d] - 1;
    } else if (dir == direction::right) {
      child_i[d] = 2 * child_i[d] + 1;
    } else {
      throw; // never
    }
  }

  inline void neighbor_pos_dir(std::vector<int64_t> &l, std::vector<int64_t> &i,
                               std::vector<double> &neighbor, direction dir) {
    if (dir == direction::left) {
      for (int64_t d = 0; d < dim; d += 1) {
        neighbor[d] = std::pow(2.0, static_cast<double>(-l[d])) *
                      static_cast<double>(i[d] - 1ll);
      }
    } else if (dir == direction::right) {
      for (int64_t d = 0; d < dim; d += 1) {
        neighbor[d] = std::pow(2.0, static_cast<double>(-l[d])) *
                      static_cast<double>(i[d] + 1ll);
      }
    } else {
      throw; // never
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
  spatial_refinement_blocked(int64_t dim, int64_t max_level,
                             int64_t min_support,
                             const std::vector<double> &data);
  void refine();
  std::vector<int64_t> &get_levels();
  std::vector<int64_t> &get_indices();
  void print_level(std::vector<int64_t> l);
  void print_index(std::vector<int64_t> i);
  void print_pos(std::vector<double> pos);
  static void write_grid_positions(int64_t dim,
                                   const std::string &grid_file_name,
                                   const std::string &dataset_file_name,
                                   std::vector<int64_t> &ls,
                                   std::vector<int64_t> &is);

#if USE_OCL == 1
  inline void enable_OCL(const std::string &ocl_config_file_name) {
    this->use_ocl = true;
    this->ocl_config_file_name = ocl_config_file_name;
    auto ocl_init_start = std::chrono::system_clock::now();
    this->manager = std::make_unique<opencl::manager_t>(ocl_config_file_name);
    auto ocl_init_stop = std::chrono::system_clock::now();
    std::chrono::duration<double> ocl_init_seconds =
        ocl_init_stop - ocl_init_start;
    double ocl_init_duration = ocl_init_seconds.count();
    std::cout << "ocl_init_duration: " << ocl_init_duration << std::endl;
    device = manager->get_devices()[0];

    if (manager->get_devices().size() > 1) {
      std::cout << "warning: support refinement: more than one device "
                   "configured, using first device only!"
                << std::endl;
      // throw opencl::manager_error("only supports single device");
    }

    if (!kernel_verify_support) {
      std::string kernel_src_file_name{
          "datadriven/src/sgpp/datadriven/grid/spatial_refinement_kernel.cl"};
      std::string kernel_src = manager->read_src_file(kernel_src_file_name);
      json::node &deviceNode =
          manager->get_configuration()["PLATFORMS"][device.platformName]
                                      ["DEVICES"][device.deviceName];
      json::node &kernelConfig = deviceNode["KERNELS"]["verify_support"];
      kernel_verify_support = manager->build_kernel(
          kernel_src, device, kernelConfig, "verify_support",
          std::string("-DDIM=") + std::to_string(dim));
    }

    int64_t num_data = static_cast<int64_t>(data.size()) / dim;
    int64_t num_data_padded = num_data;

    if (num_data % local_cache != 0) {
      num_data_padded += local_cache - (num_data % local_cache);
    }

    this->data_device = std::make_unique<opencl::managed_buffer<double>>(
        device, num_data_padded * dim);
    data_device->fill_buffer(
        -1.0); // generates out of support data points in padding
    data_device->to_device(data);
  }
#endif

}; // namespace datadriven

} // namespace sgpp::datadriven
