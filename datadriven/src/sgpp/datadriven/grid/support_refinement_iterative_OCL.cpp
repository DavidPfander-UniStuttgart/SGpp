#if USE_OCL == 1

#include "sgpp/base/opencl/manager/apply_arguments.hpp"
#include "sgpp/base/opencl/manager/managed_buffer.hpp"
#include "sgpp/base/opencl/manager/manager.hpp"
#include "sgpp/base/opencl/manager/manager_error.hpp"
#include "sgpp/base/opencl/manager/run_kernel.hpp"

#include "support_refinement_iterative.hpp"

namespace sgpp::datadriven {

void support_refinement_iterative::verify_support_blocked_OCL() {
  int64_t num_candidates = static_cast<int64_t>(schedule_level.size()) / dim;
  // int64_t blocks_total = num_candidates_padded / block_size;
  // truncated number of blocks per device, potentially larger schedule for last
  // device
  // int64_t blocks_per_device = blocks_total / manager->get_devices().size();
  int64_t num_candidates_per_device =
      num_candidates / manager->get_devices().size();

  std::cout << "num_candidates: " << num_candidates
            // << " num_candidates_padded: " << num_candidates_padded
            << " block_size: " << block_size
            << " num_candidates_per_device: " << num_candidates_per_device
            << std::endl;
// << " blocks_per_device: " << blocks_per_device << std::endl;

#pragma omp parallel for
  for (size_t i = 0; i < manager->get_devices().size(); i += 1) {
    int64_t range_start = i * num_candidates_per_device;
    int64_t range_end = (i + 1) * num_candidates_per_device; // exclusive
    // special case if there are too few work items for multiple GPUs
    if (num_candidates < 4096) {
      if (i == manager->get_devices().size() - 1) {
        range_start = 0;
      } else {
        continue;
      }
    }
    // last device gets remaining work items
    if (i == manager->get_devices().size() - 1) {
      range_end = num_candidates;
    }

    int64_t range_size = range_end - range_start;

    int64_t range_padded = range_size;
    if (range_padded % block_size != 0) {
      range_padded += block_size - (range_padded % block_size);
    }

    std::cout << "device: " << i << " range_padded: " << range_padded
              << " range_start: " << range_start << " range_end: " << range_end
              << std::endl;
    if (range_end <= 0) {
      continue;
    }

    opencl::managed_buffer<long> schedule_level_device(
        manager->get_devices()[i], range_padded * dim);
    schedule_level_device.fill_buffer(1);
    schedule_level_device.to_device(schedule_level, range_start * dim,
                                    range_end * dim); // requires range

    opencl::managed_buffer<long> schedule_index_device(
        manager->get_devices()[i], range_padded * dim);
    schedule_index_device.fill_buffer(1);
    schedule_index_device.to_device(schedule_index, range_start * dim,
                                    range_end * dim); // requires range

    opencl::managed_buffer<bool> schedule_support_device(
        manager->get_devices()[i], range_padded);
    schedule_support_device.fill_buffer(false);

    opencl::apply_arguments(kernels_verify_support[i], data_devices[i]->get(),
                            entries, schedule_level_device.get(),
                            schedule_index_device.get(),
                            schedule_support_device.get(), min_support);

    opencl::run_kernel_1d_timed(this->manager->get_devices()[i],
                                kernels_verify_support[i], range_padded,
                                block_size);

#pragma omp critical (support_refinement_multi_device)
    {
      schedule_support_device.from_device(schedule_support, range_start,
                                          range_end);
    }
  }
  // std::cout << "schedule_support: ";
  // for (size_t i = 0; i < schedule_support.size(); i += 1) {
  //   std::cout << schedule_support[i];
  // }
  // std::cout << std::endl;
}

void support_refinement_iterative::enable_OCL(
    const std::string &ocl_config_file_name) {
  if (this->use_ocl == true) {
    if (!this->manager) {
      throw;
    }
    this->manager->release_kernels(kernels_verify_support);
  }

  this->use_ocl = true;
  this->ocl_config_file_name = ocl_config_file_name;
  auto ocl_init_start = std::chrono::system_clock::now();
  this->manager = std::make_unique<opencl::manager_t>(ocl_config_file_name);
  auto ocl_init_stop = std::chrono::system_clock::now();
  std::chrono::duration<double> ocl_init_seconds =
      ocl_init_stop - ocl_init_start;
  double ocl_init_duration = ocl_init_seconds.count();
  std::cout << "ocl_init_duration: " << ocl_init_duration << std::endl;

  // compile all kernels
  std::string kernel_src_file_name{
      "datadriven/src/sgpp/datadriven/grid/support_refinement_kernel.cl"};
  std::string kernel_src = manager->read_src_file(kernel_src_file_name);
  kernels_verify_support =
      manager->build_kernel(kernel_src, "verify_support",
                            std::string("-DDIM=") + std::to_string(dim));

  int64_t num_data = static_cast<int64_t>(data.size()) / dim;
  int64_t num_data_padded = num_data;

  if (num_data % local_cache != 0) {
    num_data_padded += local_cache - (num_data % local_cache);
  }

  this->data_devices.resize(manager->get_devices().size());
  for (size_t i = 0; i < manager->get_devices().size(); i += 1) {
    data_devices[i] = std::make_unique<opencl::managed_buffer<double>>(
        manager->get_devices()[i], num_data_padded * dim);
    data_devices[i]->fill_buffer(
        -1.0); // generates out of support data points in padding
    data_devices[i]->to_device(data);
  }
}

} // namespace sgpp::datadriven
#endif
