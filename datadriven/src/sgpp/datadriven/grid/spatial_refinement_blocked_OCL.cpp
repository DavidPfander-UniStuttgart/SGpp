#if USE_OCL == 1

#include "sgpp/base/opencl/manager/apply_arguments.hpp"
#include "sgpp/base/opencl/manager/managed_buffer.hpp"
#include "sgpp/base/opencl/manager/manager.hpp"
#include "sgpp/base/opencl/manager/manager_error.hpp"
#include "sgpp/base/opencl/manager/run_kernel.hpp"

#include "spatial_refinement_blocked.hpp"

namespace sgpp::datadriven {

void spatial_refinement_blocked::verify_support_blocked_OCL() {
  std::string configuration_file(ocl_config_file_name);
  opencl::manager_t manager(configuration_file);
  if (manager.get_devices().size() > 1) {
    throw opencl::manager_error("only supports single device");
  }
  opencl::device_t device = manager.get_devices()[0];
  opencl::managed_buffer<double> data_device(device, data.size());
  data_device.to_device(data);

  const int block_size = 256;

  int64_t num_candidates = static_cast<int64_t>(schedule_level.size() / dim);
  int64_t num_candidates_padded = num_candidates;
  if (num_candidates % block_size != 0) {
    num_candidates_padded += block_size - (num_candidates % block_size);
  }

  opencl::managed_buffer<long> schedule_level_device(
      device, num_candidates_padded * dim);
  schedule_level_device.fill_buffer(1);
  schedule_level_device.to_device(schedule_level);

  opencl::managed_buffer<long> schedule_index_device(
      device, num_candidates_padded * dim);
  schedule_index_device.fill_buffer(1);
  schedule_index_device.to_device(schedule_index);

  opencl::managed_buffer<bool> schedule_support_device(device,
                                                       num_candidates_padded);
  schedule_support_device.fill_buffer(false);

  cl_kernel kernel_verify_support;
  if (!kernel_verify_support) {
    std::string kernel_src_file_name{
        "datadriven/src/sgpp/datadriven/grid/spatial_refinement_kernel.cl"};
    std::string kernel_src = manager.read_src_file(kernel_src_file_name);
    json::node &deviceNode =
        manager.get_configuration()["PLATFORMS"][device.platformName]["DEVICES"]
                                   [device.deviceName];
    json::node &kernelConfig = deviceNode["KERNELS"]["verify_support"];
    kernel_verify_support =
        manager.build_kernel(kernel_src, device, kernelConfig, "verify_support",
                             std::string("-DDIM=") + std::to_string(dim));
  }

  opencl::apply_arguments(kernel_verify_support, data_device.get(), entries,
                          schedule_level_device.get(),
                          schedule_index_device.get(),
                          schedule_support_device.get(), min_support);

  int grid_size = num_candidates_padded; // / block_size
  std::cout << "num_candidates: " << num_candidates
            << " num_candidates_padded: " << num_candidates_padded
            << " grid_size: " << grid_size << " block_size: " << block_size
            << std::endl;
  opencl::run_kernel_1d_timed(device, kernel_verify_support, grid_size,
                              block_size);
  schedule_support_device.from_device(schedule_support);
}

} // namespace sgpp::datadriven
#endif
