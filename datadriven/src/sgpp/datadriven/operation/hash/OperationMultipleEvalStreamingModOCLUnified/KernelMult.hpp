// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#pragma once

#include <CL/cl.h>
#include <omp.h>

#include <chrono>
#include <limits>
#include <string>
#include <vector>

#include "SourceBuilderMult.hpp"
#include "sgpp/base/opencl/OCLBufferWrapperSD.hpp"
#include "sgpp/base/opencl/OCLManagerMultiPlatform.hpp"
#include "sgpp/base/opencl/OCLStretchedBuffer.hpp"
#include "sgpp/base/opencl/manager/apply_arguments.hpp"
#include "sgpp/base/tools/QueueLoadBalancerOpenMP.hpp"
#include "sgpp/globaldef.hpp"

namespace sgpp {
namespace datadriven {
namespace StreamingModOCLUnified {

template <typename real_type> class KernelMult {
private:
  std::shared_ptr<base::OCLDevice> device;

  size_t dims;

  cl_int err;

  base::OCLBufferWrapperSD<real_type> deviceLevel;
  base::OCLBufferWrapperSD<real_type> deviceIndex;
  base::OCLBufferWrapperSD<real_type> deviceAlpha;

  base::OCLBufferWrapperSD<real_type> deviceData;

  base::OCLBufferWrapperSD<real_type> deviceResultData;

  cl_kernel kernelMult;

  double deviceTimingMult;

  SourceBuilderMult<real_type> kernelSourceBuilder;
  std::shared_ptr<base::OCLManagerMultiPlatform> manager;
  json::node &kernelConfiguration;

  std::shared_ptr<base::QueueLoadBalancerOpenMP> queueLoadBalancerMult;

  std::vector<real_type> &dataset;

  int64_t num_devices;

  bool verbose;

  size_t localSize;
  size_t dataBlockingSize;
  // size_t scheduleSize;
  size_t totalBlockSize;
  size_t gridSplit;
  bool transferWholeDataset;

  bool do_reset;
  // only relevant if whole dataset is transferred
  bool dataset_transferred;

  std::vector<real_type> zeros;

public:
  KernelMult(std::shared_ptr<base::OCLDevice> device, size_t dims,
             std::shared_ptr<base::OCLManagerMultiPlatform> manager,
             json::node &kernelConfiguration,
             std::shared_ptr<base::QueueLoadBalancerOpenMP> queueBalancerMult,
             std::vector<real_type> &dataset, bool isModLinear,
             int64_t num_devices)
      : device(device), dims(dims), err(CL_SUCCESS), deviceLevel(device),
        deviceIndex(device), deviceAlpha(device), deviceData(device),
        deviceResultData(device), kernelMult(nullptr),
        kernelSourceBuilder(device, kernelConfiguration, dims, isModLinear),
        manager(manager), kernelConfiguration(kernelConfiguration),
        queueLoadBalancerMult(queueBalancerMult), dataset(dataset),
        num_devices(num_devices), do_reset(true), dataset_transferred(false) {
    if (kernelConfiguration["KERNEL_STORE_DATA"].get().compare("register") ==
            0 &&
        dims > kernelConfiguration["KERNEL_MAX_DIM_UNROLL"].getUInt()) {
      std::stringstream errorString;
      errorString << "OCL Error: setting \"KERNEL_DATA_STORE\" to \"register\" "
                     "requires value of "
                     "\"KERNEL_MAX_DIM_UNROLL\" to be greater than the "
                     "dimension of the data "
                     "set, was set to "
                  << kernelConfiguration["KERNEL_MAX_DIM_UNROLL"].getUInt()
                  << std::endl;
      throw sgpp::base::operation_exception(errorString.str());
    }

    // initialize with same timing to enforce equal problem sizes in the
    // beginning
    this->deviceTimingMult = 1.0;
    this->verbose = kernelConfiguration["VERBOSE"].getBool();

    localSize = kernelConfiguration["LOCAL_SIZE"].getUInt();
    dataBlockingSize = kernelConfiguration["KERNEL_DATA_BLOCK_SIZE"].getUInt();
    // scheduleSize = kernelConfiguration["KERNEL_SCHEDULE_SIZE"].getUInt();
    totalBlockSize = localSize * dataBlockingSize;
    gridSplit = kernelConfiguration["KERNEL_GRID_SPLIT"].getUInt();
    transferWholeDataset =
        kernelConfiguration["KERNEL_TRANSFER_WHOLE_DATASET"].getBool();
  }

  ~KernelMult() {
    if (this->kernelMult != nullptr) {
      clReleaseKernel(this->kernelMult);
      this->kernelMult = nullptr;
    }
  }

  void resetKernel() { do_reset = true; }

  double mult(std::vector<real_type> &level, std::vector<real_type> &index,
              std::vector<real_type> &alpha, // std::vector<real_type> &dataset,
              std::vector<real_type> &result, const size_t start_index_grid,
              const size_t end_index_grid, const size_t start_index_data,
              const size_t end_index_data) {
    // check if there is something to do at all
    if (!(end_index_grid > start_index_grid &&
          end_index_data > start_index_data)) {
      return 0.0;
    }

    if (this->kernelMult == nullptr) {
      std::string program_src = kernelSourceBuilder.generateSource();
      this->kernelMult = manager->buildKernel(
          program_src, device, kernelConfiguration, "multOCLUnified");
    }

    if (do_reset) {
      deviceLevel.intializeTo(level, dims, start_index_grid, end_index_grid);
      deviceIndex.intializeTo(index, dims, start_index_grid, end_index_grid);
      do_reset = false;
    }
    if (transferWholeDataset && !dataset_transferred) {
      deviceData.intializeTo(dataset, dims, start_index_data, end_index_data,
                             true);
      dataset_transferred = true;
    }
    deviceAlpha.intializeTo(alpha, 1, start_index_grid, end_index_grid);
    // clFinish(device->commandQueue);

    this->deviceTimingMult = 0.0;

    // configure schedule size for a single valid-sized block per device
    // raw schedule size
    size_t scheduleSize = queueLoadBalancerMult->getRange() / num_devices;
    // make divisible by localSize*blockSize
    if (scheduleSize % totalBlockSize != 0) {
      scheduleSize += totalBlockSize - (scheduleSize % totalBlockSize);
    }

    while (true) {
      size_t kernelStartData = 0;
      size_t kernelEndData = 0;
      bool segmentAvailable = queueLoadBalancerMult->getNextSegment(
          scheduleSize, kernelStartData, kernelEndData);
      if (!segmentAvailable) {
        break;
      }

      size_t rangeSizeUnblocked = kernelEndData - kernelStartData;

      if (verbose) {
        std::cout << "device: " << device->deviceId
                  << " kernel from: " << kernelStartData
                  << " to: " << kernelEndData
                  << " -> range: " << rangeSizeUnblocked << std::endl;
      }

      // clFinish(device->commandQueue);
      // std::chrono::time_point<std::chrono::system_clock> start, end;
      // start = std::chrono::system_clock::now();

      // transfer partial dataset every iteration, for large datasets
      if (!transferWholeDataset) {
        deviceData.intializeTo(dataset, dims, kernelStartData, kernelEndData,
                               true);
      }
      size_t range = kernelEndData - kernelStartData;
      if (zeros.size() != range) {
        zeros.resize(range);
        for (size_t i = 0; i < range; i++) {
          zeros[i] = 0.0;
        }
      }
      deviceResultData.intializeTo(zeros, 1, 0, range);
      clFinish(device->commandQueue);

      // end = std::chrono::system_clock::now();
      // std::chrono::duration<double> elapsed_seconds = end - start;
      // if (verbose) {
      //   std::cout << "init buffers mult: " << elapsed_seconds.count()
      //             << std::endl;
      // }

      size_t rangeSizeBlocked = (kernelEndData / dataBlockingSize) -
                                (kernelStartData / dataBlockingSize);

      if (rangeSizeBlocked > 0) {
        // assuming transferring whole dataset
        int deviceDataSize = end_index_data - start_index_data;
        int deviceDataOffset = kernelStartData;
        if (!transferWholeDataset) {
          deviceDataSize = kernelEndData - kernelStartData;
          deviceDataOffset = 0;
        }

        opencl::apply_arguments(
            this->kernelMult, *(this->deviceLevel.getBuffer()),
            *(this->deviceIndex.getBuffer()), *(this->deviceData.getBuffer()),
            *(this->deviceAlpha.getBuffer()),
            *(this->deviceResultData.getBuffer()),
            static_cast<int>(rangeSizeUnblocked), deviceDataSize,
            deviceDataOffset, static_cast<int>(start_index_grid),
            static_cast<int>(end_index_grid));

        cl_event clTiming = nullptr;

        // clFinish(device->commandQueue);

        const size_t rangeSizeBlocked2D[2] = {rangeSizeBlocked, gridSplit};
        const size_t localSize2D[2] = {localSize, 1};
        err = clEnqueueNDRangeKernel(device->commandQueue, this->kernelMult, 2,
                                     0, rangeSizeBlocked2D, localSize2D, 0,
                                     nullptr, &clTiming);

        if (err != CL_SUCCESS) {
          std::stringstream errorString;
          errorString
              << "OCL Error: Failed to enqueue kernel command! Error code: "
              << err << std::endl;
          throw sgpp::base::operation_exception(errorString.str());
        }

        clFinish(device->commandQueue);

        deviceResultData.readFromBuffer();

        clFinish(device->commandQueue);

        std::vector<real_type> &hostTemp = deviceResultData.getHostPointer();
        size_t deviceIndex = 0;
        for (size_t i = 0; i < rangeSizeUnblocked; i++) {
          result[kernelStartData + i] = hostTemp[deviceIndex];
          deviceIndex += 1;
        }

        // determine kernel execution time
        cl_ulong startTime = 0;
        cl_ulong endTime = 0;

        err = clGetEventProfilingInfo(clTiming, CL_PROFILING_COMMAND_START,
                                      sizeof(cl_ulong), &startTime, nullptr);

        if (err != CL_SUCCESS) {
          std::stringstream errorString;
          errorString << "OCL Error: Failed to read start-time from command "
                         "queue (or crash in mult)! Error code: "
                      << err << std::endl;
          throw sgpp::base::operation_exception(errorString.str());
        }

        err = clGetEventProfilingInfo(clTiming, CL_PROFILING_COMMAND_END,
                                      sizeof(cl_ulong), &endTime, nullptr);

        if (err != CL_SUCCESS) {
          std::stringstream errorString;
          errorString << "OCL Error: Failed to read end-time from command "
                         "queue! Error code: "
                      << err << std::endl;
          throw sgpp::base::operation_exception(errorString.str());
        }

        clReleaseEvent(clTiming);

        double time = 0.0;
        time = static_cast<double>(endTime - startTime);
        time *= 1e-9;

        if (verbose) {
          std::cout << "device: " << device->deviceId << " duration: " << time
                    << std::endl;
        }

        this->deviceTimingMult += time;
      }
    }

    return this->deviceTimingMult;
  }
};
} // namespace StreamingModOCLUnified
} // namespace datadriven
} // namespace sgpp
