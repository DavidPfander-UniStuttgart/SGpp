// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org
#pragma once

#include <CL/cl.h>

#include <sgpp/base/opencl/OCLBufferWrapperSD.hpp>
#include <sgpp/base/opencl/OCLManagerMultiPlatform.hpp>
#include <sgpp/base/opencl/OCLOperationConfiguration.hpp>
#include <sgpp/globaldef.hpp>

#include <limits>
#include <string>
#include <vector>

#include "KernelSourceBuilderCreateGraph.hpp"

namespace sgpp {
namespace datadriven {
namespace DensityOCLMultiPlatform {

/// OpenCL kernel class for operation to create a k nearest neighbor graph
template <typename T>
class KernelCreateGraph {
 private:
  /// Used opencl device
  std::shared_ptr<base::OCLDevice> device;
  size_t dims;
  size_t k;
  cl_int err;
  /// OpenCL buffer for the dataset
  base::OCLBufferWrapperSD<T> deviceData;
  /// OpenCL Buffer for the result vector
  base::OCLBufferWrapperSD<int64_t> deviceResultData;
  cl_kernel kernel;
  /// Source builder for the opencl source code of the kernel
  sgpp::datadriven::DensityOCLMultiPlatform::SourceBuilderCreateGraph<T> kernelSourceBuilder;
  /// OpenCL manager
  std::shared_ptr<base::OCLManagerMultiPlatform> manager;
  /// Stores the running time of the kernel
  double deviceTimingMult;
  bool verbose;
  /// OpenCL configuration containing the building flags
  json::Node &kernelConfiguration;
  size_t localSize;
  /// Host side buffer for the dataset
  std::vector<T> &data;
  size_t unpadded_data_size;
  size_t padded_data_size;
  cl_event clTiming = nullptr;

 public:
  KernelCreateGraph(std::shared_ptr<base::OCLDevice> dev, size_t dims, size_t k,
                    std::vector<T> &data, std::shared_ptr<base::OCLManagerMultiPlatform> manager,
                    json::Node &kernelConfiguration)
      : device(dev),
        dims(dims),
        k(k),
        err(CL_SUCCESS),
        deviceData(device),
        deviceResultData(device),
        kernel(nullptr),
        kernelSourceBuilder(kernelConfiguration, dims),
        manager(manager),
        deviceTimingMult(0.0),
        verbose(false),
        kernelConfiguration(kernelConfiguration),
        data(data) {
    this->verbose = kernelConfiguration["VERBOSE"].getBool();

    localSize = kernelConfiguration["LOCAL_SIZE"].getUInt();

    if (kernelConfiguration.contains("APPROX_REG_COUNT")) {
      size_t approxRegCount = kernelConfiguration["APPROX_REG_COUNT"].getUInt();
      // Check range and whether x is a power of 2 or not
      if (approxRegCount < k || approxRegCount > localSize ||
          (approxRegCount & (approxRegCount - 1))) {
        std::stringstream errorString;
        errorString << "OCL Error: APPROX_REG_COUNT: " << approxRegCount
                    << " is not a valid size!\n"
                    << "Needs to be a power of 2, greater than k and smaller than (or equal to) the"
                    << " parameter LOCAL_SIZE: " << localSize;
        throw base::operation_exception(errorString.str());
      }
    }

    unpadded_data_size = data.size() / dims;
    size_t element_to_add = localSize - (unpadded_data_size % localSize);
    padded_data_size = unpadded_data_size + element_to_add;
    std::cout << "unpadded_datasize: " << unpadded_data_size
              << " adding elements: " << element_to_add << " (* dims)" << std::endl;
    // max difference between valid elements (squared): dims
    double padd_value = 3.0 * dims;
    for (size_t i = 0; i < element_to_add * dims; i++) {
      data.push_back(padd_value);
    }
    deviceData.intializeTo(data, 1, 0, data.size());
  }

  ~KernelCreateGraph() {
    if (kernel != nullptr) {
      clReleaseKernel(kernel);
      this->kernel = nullptr;
    }
  }

  /// Runs the opencl kernel to find the k nearest neighbors of all datapoints in the given chunk
  void begin_graph_creation(size_t startid, size_t chunksize) {
    if (verbose) {
      std::cout << "entering graph, device: " << device->deviceName << " (" << device->deviceId
                << ")" << std::endl;
      std::cout << "k: " << k << " Dims:" << dims << std::endl;
    }

    size_t globalworkrange[1];
    if (chunksize == 0) {
      globalworkrange[0] = padded_data_size;
    } else {
      globalworkrange[0] = chunksize;
      size_t element_to_add = localSize - (chunksize % localSize);
      globalworkrange[0] += element_to_add;
    }

    // Build kernel if not already done
    if (this->kernel == nullptr) {
      if (verbose) std::cout << "generating kernel source" << std::endl;
      std::string program_src = kernelSourceBuilder.generateSource(
          dims, k, padded_data_size, padded_data_size
          // (unpadded_datasize / dims) + (localSize - (unpadded_datasize / dims) % localSize)
      );
      // if (verbose) std::cout << "Source: " << std::endl << program_src << std::endl;
      if (verbose) std::cout << "building kernel" << std::endl;
      this->kernel =
          manager->buildKernel(program_src, device, kernelConfiguration, "connectNeighbors");
    }

    if (!deviceResultData.isInitialized() || deviceResultData.size() < globalworkrange[0] * k) {
      deviceResultData.initializeBuffer(globalworkrange[0] * k);
    }
    clFinish(device->commandQueue);
    this->deviceTimingMult = 0.0;

    // Set kernel arguments
    err = clSetKernelArg(this->kernel, 0, sizeof(cl_mem), this->deviceData.getBuffer());
    if (err != CL_SUCCESS) {
      std::stringstream errorString;
      errorString << "OCL Error: Failed to create kernel arguments for device " << std::endl;
      throw base::operation_exception(errorString.str());
    }
    err = clSetKernelArg(this->kernel, 1, sizeof(cl_mem), this->deviceResultData.getBuffer());
    if (err != CL_SUCCESS) {
      std::stringstream errorString;
      errorString << "OCL Error: Failed to create kernel arguments for device " << std::endl;
      throw base::operation_exception(errorString.str());
    }
    err = clSetKernelArg(this->kernel, 2, sizeof(cl_ulong), &startid);
    if (err != CL_SUCCESS) {
      std::stringstream errorString;
      errorString << "OCL Error: Failed to create kernel arguments for device " << std::endl;
      throw base::operation_exception(errorString.str());
    }

    clTiming = nullptr;

    if (verbose) {
      std::cout << "Starting the kernel for " << globalworkrange[0] << " items" << std::endl;
    }
    err = clEnqueueNDRangeKernel(device->commandQueue, this->kernel, 1, 0, globalworkrange,
                                 &localSize, 0, nullptr, &clTiming);
    if (err != CL_SUCCESS) {
      std::stringstream errorString;
      errorString << "OCL Error: Failed to enqueue kernel command! Error code: " << err
                  << std::endl;
      throw base::operation_exception(errorString.str());
    }
  }
  double finalize_graph_creation(std::vector<long> &result, size_t startid, size_t chunksize) {
    clFinish(device->commandQueue);

    if (verbose) {
      std::cout << "Finished kernel execution" << std::endl;
    }
    deviceResultData.readFromBuffer();
    clFinish(device->commandQueue);

    std::vector<int64_t> &hostTemp = deviceResultData.getHostPointer();

    size_t real_count;
    if (chunksize == 0) {
      // real_count = unpadded_data_size / dims;
      real_count = unpadded_data_size;
    } else {
      real_count = chunksize;
    }
    for (size_t i = 0; i < real_count * k; i++) {
      result[i] = hostTemp[i];
    }
    if (verbose) {
      std::cout << "Read data from opencl device" << std::endl;
    }
    // determine kernel execution time
    cl_ulong startTime = 0;
    cl_ulong endTime = 0;

    err = clGetEventProfilingInfo(clTiming, CL_PROFILING_COMMAND_START, sizeof(cl_ulong),
                                  &startTime, nullptr);

    if (err != CL_SUCCESS) {
      std::stringstream errorString;
      errorString << "OCL Error: Failed to read start-time from command queue "
                  << "(or crash in mult)! Error code: " << err << std::endl;
      throw base::operation_exception(errorString.str());
    }

    err = clGetEventProfilingInfo(clTiming, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime,
                                  nullptr);

    if (err != CL_SUCCESS) {
      std::stringstream errorString;
      errorString << "OCL Error: Failed to read end-time from command queue! "
                  << " Error code: " << err << std::endl;
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
  /// Adds all possible building parameters to the configuration if they do not exist yet
  static void augmentDefaultParameters(sgpp::base::OCLOperationConfiguration &parameters) {
    for (std::string &platformName : parameters["PLATFORMS"].keys()) {
      json::Node &platformNode = parameters["PLATFORMS"][platformName];
      for (std::string &deviceName : platformNode["DEVICES"].keys()) {
        json::Node &deviceNode = platformNode["DEVICES"][deviceName];

        const std::string &kernelName = "connectNeighbors";

        json::Node &kernelNode = deviceNode["KERNELS"].contains(kernelName)
                                     ? deviceNode["KERNELS"][kernelName]
                                     : deviceNode["KERNELS"].addDictAttr(kernelName);

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

        if (kernelNode.contains("USE_SELECT") == false) {
          kernelNode.addIDAttr("USE_SELECT", false);
        }

        if (kernelNode.contains("USE_APPROX") == false) {
          kernelNode.addIDAttr("USE_APPROX", false);
        }
      }
    }
  }
};

}  // namespace DensityOCLMultiPlatform
}  // namespace datadriven
}  // namespace sgpp
