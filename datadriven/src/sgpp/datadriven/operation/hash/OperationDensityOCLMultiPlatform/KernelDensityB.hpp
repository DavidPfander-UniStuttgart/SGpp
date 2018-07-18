// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org
#pragma once

#include <sgpp/base/grid/storage/compressed/CompressedGrid.hpp>
#include <sgpp/base/opencl/OCLBufferWrapperSD.hpp>
#include <sgpp/base/opencl/OCLManagerMultiPlatform.hpp>
#include <sgpp/base/opencl/OCLOperationConfiguration.hpp>
#include <sgpp/globaldef.hpp>

#include <CL/cl.h>
#include <string.h>
#include <limits>
#include <string>
#include <vector>

#include "KernelSourceBuilderB.hpp"

namespace sgpp {
namespace datadriven {
namespace DensityOCLMultiPlatform {

/// Class for the OpenCL density matrix vector multiplication
template <typename T>
class KernelDensityB {
 private:
  /// Used opencl device
  std::shared_ptr<base::OCLDevice> device;

  size_t dims;
  size_t dataSize;
  size_t gridSize;

  cl_int err;

  /// Buffer for the grid points
  base::OCLBufferWrapperSD<int> devicePoints;
  /// Buffer for the used dataset
  base::OCLBufferWrapperSD<T> deviceData;
  /// Buffer for the result vector
  base::OCLBufferWrapperSD<T> deviceResultData;
  // Compression buffers
  base::OCLBufferWrapperSD<unsigned long> device_dim_zero_flags;
  base::OCLBufferWrapperSD<unsigned long> device_level_offsets;
  base::OCLBufferWrapperSD<unsigned long> device_level_packed;
  base::OCLBufferWrapperSD<unsigned long> device_index_packed;

  cl_kernel kernelB;

  /// Source builder for the kernel opencl source code
  sgpp::datadriven::DensityOCLMultiPlatform::SourceBuilderB<T> kernelSourceBuilder;
  /// OpenCL manager
  std::shared_ptr<base::OCLManagerMultiPlatform> manager;

  double deviceTimingMult;

  /// OpenCL configuration containing the building flags
  json::Node &kernelConfiguration;

  bool verbose;

  cl_event clTiming = nullptr;

  bool use_compression;
  size_t localSize;

 public:
  KernelDensityB(std::shared_ptr<base::OCLDevice> dev, size_t dims,
                 std::shared_ptr<base::OCLManagerMultiPlatform> manager,
                 json::Node &kernelConfiguration, std::vector<int> &points)
      : device(dev),
        dims(dims),
        err(CL_SUCCESS),
        devicePoints(device),
        deviceData(device),
        deviceResultData(device),
        device_dim_zero_flags(device),
        device_level_offsets(device),
        device_level_packed(device),
        device_index_packed(device),
        kernelB(nullptr),
        kernelSourceBuilder(kernelConfiguration, dims),
        manager(manager),
        deviceTimingMult(0.0),
        use_compression(false),
        kernelConfiguration(kernelConfiguration) {
    gridSize = points.size() / (2 * dims);
    this->verbose = kernelConfiguration["VERBOSE"].getBool();

    localSize = kernelConfiguration["LOCAL_SIZE"].getUInt();

    if (kernelConfiguration.contains("USE_COMPRESSION_FIXED")) {
      use_compression = kernelConfiguration["USE_COMPRESSION_FIXED"].getBool();
      compressed_grid grid(points, dims);
      // if(!grid.check_grid_compression(points)) {
      //   std::cerr << "Grid compression check failed! " << std::endl;
      // } else {
      //   std::cerr << "Grid compression check succeded! " << std::endl;
      // }
      device_dim_zero_flags.intializeTo(grid.dim_zero_flags_v, 1, 0, grid.dim_zero_flags_v.size());
      device_level_offsets.intializeTo(grid.level_offsets_v, 1, 0, grid.level_offsets_v.size());
      device_level_packed.intializeTo(grid.level_packed_v, 1, 0, grid.level_packed_v.size());
      device_index_packed.intializeTo(grid.index_packed_v, 1, 0, grid.index_packed_v.size());
    } else {
      devicePoints.intializeTo(points, 1, 0, points.size());
    }
  }

  ~KernelDensityB() {
    if (kernelB != nullptr) {
      clReleaseKernel(kernelB);
      this->kernelB = nullptr;
    }
  }

  void initialize_dataset(std::vector<T> &data) {
    this->dataSize = data.size() / dims;
    if (!deviceData.isInitialized()) deviceData.intializeTo(data, 1, 0, data.size());
  }

  /// Generates part of the right hand side density vector
  void start_rhs_generation(size_t startid = 0, size_t chunksize = 0) {
    if (verbose) {
      std::cout << "entering mult, device: " << device->deviceName << " (" << device->deviceId
                << ")" << std::endl;
    }

    // Build kernel if not already done
    if (this->kernelB == nullptr) {
      if (verbose) std::cout << "generating kernel source" << std::endl;
      std::string program_src = kernelSourceBuilder.generateSource(dims, dataSize, gridSize);
      // if (verbose) std::cout << "Source: " << std::endl << program_src << std::endl;
      if (verbose) std::cout << "building kernel" << std::endl;
      std::cout << std::flush;
      this->kernelB = manager->buildKernel(program_src, device, kernelConfiguration, "cscheme");
    }

    // Load data into buffers if not already done
    if (!deviceResultData.isInitialized()) {
      if (chunksize == 0) {
        std::vector<T> zeros(gridSize);
        for (size_t i = 0; i < gridSize; i++) {
          zeros[i] = 0.0;
        }
        deviceResultData.intializeTo(zeros, 1, 0, gridSize);
      } else {
        std::vector<T> zeros(chunksize);
        for (size_t i = 0; i < chunksize; i++) {
          zeros[i] = 0.0;
        }
        deviceResultData.intializeTo(zeros, 1, 0, chunksize);
      }
      clFinish(device->commandQueue);
    }
    this->deviceTimingMult = 0.0;

    // Set kernel arguments
    size_t argument_counter = 0;
    if (!use_compression) {
      err = clSetKernelArg(this->kernelB, argument_counter, sizeof(cl_mem), this->devicePoints.getBuffer());
      if (err != CL_SUCCESS) {
        std::stringstream errorString;
        errorString << "OCL Error: Failed to create kernel arguments for device " << std::endl;
        throw base::operation_exception(errorString.str());
    }
    argument_counter++;
    } else {        err = clSetKernelArg(kernelB, argument_counter, sizeof(cl_mem),
                             this->device_dim_zero_flags.getBuffer());
        if (err != CL_SUCCESS) {
          std::stringstream errorString;
          errorString << "OCL Error: Failed to create kernel arguments (argument " << argument_counter
                      << ") for device " << std::endl;
          throw base::operation_exception(errorString.str());
        }
        argument_counter++;
        err = clSetKernelArg(kernelB, argument_counter, sizeof(cl_mem),
                             this->device_level_offsets.getBuffer());
        if (err != CL_SUCCESS) {
          std::stringstream errorString;
          errorString << "OCL Error: Failed to create kernel arguments (argument " << argument_counter
                      << ") for device " << std::endl;
          throw base::operation_exception(errorString.str());
        }
        argument_counter++;
        err = clSetKernelArg(kernelB, argument_counter, sizeof(cl_mem),
                             this->device_level_packed.getBuffer());
        if (err != CL_SUCCESS) {
          std::stringstream errorString;
          errorString << "OCL Error: Failed to create kernel arguments (argument " << argument_counter
                      << ") for device " << std::endl;
          throw base::operation_exception(errorString.str());
        }
        argument_counter++;
        err = clSetKernelArg(kernelB, argument_counter, sizeof(cl_mem),
                             this->device_index_packed.getBuffer());
        if (err != CL_SUCCESS) {
          std::stringstream errorString;
          errorString << "OCL Error: Failed to create kernel arguments (argument " << argument_counter
                      << ") for device " << std::endl;
          throw base::operation_exception(errorString.str());
        }
        argument_counter++;
    }
    err = clSetKernelArg(this->kernelB, argument_counter, sizeof(cl_mem), this->deviceData.getBuffer());
    if (err != CL_SUCCESS) {
      std::stringstream errorString;
      errorString << "OCL Error: Failed to create kernel arguments for device " << std::endl;
      throw base::operation_exception(errorString.str());
    }
    argument_counter++;
    err = clSetKernelArg(this->kernelB, argument_counter, sizeof(cl_mem), this->deviceResultData.getBuffer());
    if (err != CL_SUCCESS) {
      std::stringstream errorString;
      errorString << "OCL Error: Failed to create kernel arguments for device " << std::endl;
      throw base::operation_exception(errorString.str());
    }
    argument_counter++;
    err = clSetKernelArg(this->kernelB, argument_counter, sizeof(cl_uint), &startid);
    if (err != CL_SUCCESS) {
      std::stringstream errorString;
      errorString << "OCL Error: Failed to create kernel arguments for device " << std::endl;
      throw base::operation_exception(errorString.str());
    }
    argument_counter++;

    clTiming = nullptr;

    size_t local_size = kernelConfiguration["LOCAL_SIZE"].getUInt();

    size_t globalworkrange;
    size_t local_padding;
    if (chunksize == 0) {
      local_padding = local_size - (gridSize % local_size);
      globalworkrange = gridSize + local_padding;
    } else {
      local_padding = local_size - (chunksize % local_size);
      globalworkrange = chunksize + local_padding;
    }
    // enqueue kernel

    if (verbose) {
      std::cout << "Starting the kernel with " << globalworkrange
                << " workitems (localSize: " << local_size << ", local_padding: " << local_padding
                << ")" << std::endl;
    }

    err = clEnqueueNDRangeKernel(device->commandQueue, this->kernelB, 1, 0, &globalworkrange,
                                 &local_size, 0, nullptr, &clTiming);
    if (err != CL_SUCCESS) {
      std::stringstream errorString;
      errorString << "OCL Error: Failed to enqueue kernel command! Error code: " << err
                  << std::endl;
      throw base::operation_exception(errorString.str());
    }
  }

  double finalize_rhs_generation(std::vector<T> &result, size_t startid = 0, size_t chunksize = 0) {
    clFinish(device->commandQueue);

    if (verbose) {
      std::cout << "Finished kernel execution" << std::endl;
    }
    deviceResultData.readFromBuffer();
    clFinish(device->commandQueue);

    std::vector<T> &hostTemp = deviceResultData.getHostPointer();
    if (chunksize == 0) {
      for (size_t i = 0; i < gridSize; i++) result[i] = hostTemp[i];
    } else {
      for (size_t i = 0; i < chunksize; i++) result[i] = hostTemp[i];
    }

    // determine kernel execution time
    cl_ulong startTime = 0;
    cl_ulong endTime = 0;

    err = clGetEventProfilingInfo(clTiming, CL_PROFILING_COMMAND_START, sizeof(cl_ulong),
                                  &startTime, nullptr);

    if (err != CL_SUCCESS) {
      std::stringstream errorString;
      errorString << "OCL Error: Failed to read start-time from command queue (or crash in mult)!"
                  << " Error code: " << err << std::endl;
      throw base::operation_exception(errorString.str());
    }

    err = clGetEventProfilingInfo(clTiming, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime,
                                  nullptr);

    if (err != CL_SUCCESS) {
      std::stringstream errorString;
      errorString << "OCL Error: Failed to read end-time from command queue! Error code: " << err
                  << std::endl;
      throw base::operation_exception(errorString.str());
    }

    clReleaseEvent(clTiming);

    double time = 0.0;
    time = static_cast<double>(endTime - startTime);
    time *= 1e-9;

    if (verbose) {
      std::cout << "device: " << device->deviceName << " (" << device->deviceId << ") "
                << "duration: " << time << std::endl;
    }

    this->deviceTimingMult += time;
    return 0;
  }
  /// Adds the possible building parameters to the configuration if they do not exist yet
  static void augmentDefaultParameters(sgpp::base::OCLOperationConfiguration &parameters) {
    for (std::string &platformName : parameters["PLATFORMS"].keys()) {
      json::Node &platformNode = parameters["PLATFORMS"][platformName];
      for (std::string &deviceName : platformNode["DEVICES"].keys()) {
        json::Node &deviceNode = platformNode["DEVICES"][deviceName];

        const std::string &kernelName = "cscheme";

        json::Node &kernelNode = deviceNode["KERNELS"].contains(kernelName)
                                     ? deviceNode["KERNELS"][kernelName]
                                     : deviceNode["KERNELS"].addDictAttr(kernelName);

        if (kernelNode.contains("REUSE_SOURCE") == false) {
          kernelNode.addIDAttr("REUSE_SOURCE", false);
        }

        if (kernelNode.contains("WRITE_SOURCE") == false) {
          kernelNode.addIDAttr("WRITE_SOURCE", false);
        }

        if (kernelNode.contains("VERBOSE") == false) {
          kernelNode.addIDAttr("VERBOSE", false);
        }

        if (kernelNode.contains("LOCAL_SIZE") == false) {
          kernelNode.addIDAttr("LOCAL_SIZE", UINT64_C(128));
        }

        if (kernelNode.contains("KERNEL_USE_LOCAL_MEMORY") == false) {
          kernelNode.addIDAttr("KERNEL_USE_LOCAL_MEMORY", false);
        }

        if (kernelNode.contains("KERNEL_LOCAL_CACHE_SIZE") == false) {
          kernelNode.addIDAttr("KERNEL_LOCAL_CACHE_SIZE", UINT64_C(32));
        }
      }
    }
  }
};

}  // namespace DensityOCLMultiPlatform
}  // namespace datadriven
}  // namespace sgpp
