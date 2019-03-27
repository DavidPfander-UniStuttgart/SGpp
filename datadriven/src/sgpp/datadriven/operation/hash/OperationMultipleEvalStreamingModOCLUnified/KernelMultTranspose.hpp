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

#include "SourceBuilderMultTranspose.hpp"
#include "sgpp/base/opencl/OCLBufferWrapperSD.hpp"
#include "sgpp/base/opencl/OCLManagerMultiPlatform.hpp"
#include "sgpp/base/opencl/OCLStretchedBuffer.hpp"
#include "sgpp/base/opencl/manager/apply_arguments.hpp"
#include "sgpp/base/tools/QueueLoadBalancerOpenMP.hpp"
#include "sgpp/globaldef.hpp"

namespace sgpp {
namespace datadriven {
namespace StreamingModOCLUnified {

template <typename real_type> class KernelMultTranspose {
private:
  std::shared_ptr<base::OCLDevice> device;

  size_t dims;

  cl_int err;

  base::OCLBufferWrapperSD<real_type> deviceLevelTranspose;
  base::OCLBufferWrapperSD<real_type> deviceIndexTranspose;

  base::OCLBufferWrapperSD<real_type> deviceDataTranspose;
  base::OCLBufferWrapperSD<real_type> deviceSourceTranspose;

  base::OCLBufferWrapperSD<real_type> deviceResultGridTranspose;

  cl_kernel kernelMultTranspose;

  double deviceTimingMultTranspose;

  SourceBuilderMultTranspose<real_type> kernelSourceBuilder;
  std::shared_ptr<base::OCLManagerMultiPlatform> manager;
  //    std::shared_ptr<base::OCLOperationConfiguration> parameters;
  json::node &kernelConfiguration;

  std::shared_ptr<base::QueueLoadBalancerOpenMP> queueLoadBalancerMultTranspose;

  std::vector<real_type> &dataset;

  int64_t num_devices;

  bool verbose;

  size_t localSize;
  size_t transGridBlockingSize;
  // size_t scheduleSize;
  size_t totalBlockSize;
  size_t dataSplit;
  bool transferWholeGrid;

  bool do_reset;

  // only relevant if whole grid is transferred
  bool grid_transferred;

  std::vector<real_type> zeros;

public:
  KernelMultTranspose(
      std::shared_ptr<base::OCLDevice> device, size_t dims,
      std::shared_ptr<base::OCLManagerMultiPlatform> manager,
      json::node &kernelConfiguration,
      std::shared_ptr<base::QueueLoadBalancerOpenMP> queueBalancerMultTranpose,
      std::vector<real_type> &dataset, int64_t num_devices)
      : device(device), dims(dims), err(CL_SUCCESS),
        deviceLevelTranspose(device), deviceIndexTranspose(device),
        deviceDataTranspose(device), deviceSourceTranspose(device),
        deviceResultGridTranspose(device), kernelMultTranspose(nullptr),
        kernelSourceBuilder(device, kernelConfiguration, dims),
        manager(manager), kernelConfiguration(kernelConfiguration),
        queueLoadBalancerMultTranspose(queueBalancerMultTranpose),
        dataset(dataset), num_devices(num_devices), do_reset(true),
        grid_transferred(false) {
    if (kernelConfiguration["KERNEL_TRANS_STORE_DATA"].get().compare(
            "register") == 0 &&
        dims > kernelConfiguration["KERNEL_TRANS_MAX_DIM_UNROLL"].getUInt()) {
      std::stringstream errorString;
      errorString
          << "OCL Error: setting \"KERNEL_TRANS_DATA_STORE\" to \"register\" "
             "requires value of "
             "\"KERNEL_TRANS_MAX_DIM_UNROLL\" to be greater than the "
             "dimension of the data "
             "set, was set to "
          << kernelConfiguration["KERNEL_TRANS_MAX_DIM_UNROLL"].getUInt()
          << std::endl;
      throw sgpp::base::operation_exception(errorString.str());
    }

    // initialize with same timing to enforce equal problem sizes in the
    // beginning
    this->deviceTimingMultTranspose = 1.0;
    this->verbose = kernelConfiguration["VERBOSE"].getBool();

    localSize = kernelConfiguration["TRANS_LOCAL_SIZE"].getUInt();
    transGridBlockingSize =
        kernelConfiguration["KERNEL_TRANS_GRID_BLOCK_SIZE"].getUInt();
    // scheduleSize =
    // kernelConfiguration["KERNEL_TRANS_SCHEDULE_SIZE"].getUInt();
    totalBlockSize = localSize * transGridBlockingSize;
    dataSplit = kernelConfiguration["KERNEL_TRANS_DATA_SPLIT"].getUInt();
    transferWholeGrid =
        kernelConfiguration["KERNEL_TRANS_TRANSFER_WHOLE_GRID"].getBool();
  }

  ~KernelMultTranspose() {
    if (this->kernelMultTranspose != nullptr) {
      clReleaseKernel(this->kernelMultTranspose);
      this->kernelMultTranspose = nullptr;
    }
  }

  void resetKernel() {
    do_reset = true;
    grid_transferred = false;
  }

  double
  multTranspose(std::vector<real_type> &level, std::vector<real_type> &index,
                std::vector<real_type> &source, std::vector<real_type> &result,
                const size_t start_index_grid, const size_t end_index_grid,
                const size_t start_index_data, const size_t end_index_data) {
    // check if there is something to do at all
    if (!(end_index_grid > start_index_grid &&
          end_index_data > start_index_data)) {
      return 0.0;
    }

    if (this->kernelMultTranspose == nullptr) {
      std::string program_src = kernelSourceBuilder.generateSource();
      this->kernelMultTranspose = manager->buildKernel(
          program_src, device, kernelConfiguration, "multTransOCLUnified");
    }

    if (do_reset) {
      deviceDataTranspose.intializeTo(dataset, dims, start_index_data,
                                      end_index_data, true);
      do_reset = false;
    }
    if (transferWholeGrid && !grid_transferred) {
      deviceLevelTranspose.intializeTo(level, dims, start_index_grid,
                                       end_index_grid);
      deviceIndexTranspose.intializeTo(index, dims, start_index_grid,
                                       end_index_grid);
      grid_transferred = true;
    }
    deviceSourceTranspose.intializeTo(source, 1, start_index_data,
                                      end_index_data);
    // clFinish(device->commandQueue);

    this->deviceTimingMultTranspose = 0.0;

    // configure schedule size for a single valid-sized block per device
    // raw schedule size
    size_t scheduleSize =
        queueLoadBalancerMultTranspose->getRange() / num_devices;
    // make divisible by localSize*blockSize
    if (scheduleSize % totalBlockSize != 0) {
      scheduleSize += totalBlockSize - (scheduleSize % totalBlockSize);
    }

    while (true) {
      size_t kernelStartGrid = 0;
      size_t kernelEndGrid = 0;
      bool segmentAvailable = queueLoadBalancerMultTranspose->getNextSegment(
          scheduleSize, kernelStartGrid, kernelEndGrid);
      if (!segmentAvailable) {
        break;
      }

      size_t rangeSizeUnblocked = kernelEndGrid - kernelStartGrid;

      if (verbose) {
        // #pragma omp critical(StreamingModOCLUnifiedKernelMultTranspose)
        {
          std::cout << "device: " << device->deviceName << " ("
                    << device->deviceId << ") "
                    << " kernel from: " << kernelStartGrid
                    << " to: " << kernelEndGrid
                    << " -> range: " << rangeSizeUnblocked
                    << " (with blocking: "
                    << (rangeSizeUnblocked / this->transGridBlockingSize) << ")"
                    << std::endl;
        }
      }

      // clFinish(device->commandQueue);
      // std::chrono::time_point<std::chrono::system_clock> start, end;
      // start = std::chrono::system_clock::now();

      if (!transferWholeGrid) {
        deviceLevelTranspose.intializeTo(level, dims, kernelStartGrid,
                                         kernelEndGrid);
        deviceIndexTranspose.intializeTo(index, dims, kernelStartGrid,
                                         kernelEndGrid);
      }
      if (zeros.size() != rangeSizeUnblocked) {
        zeros.resize(rangeSizeUnblocked);
        for (size_t i = 0; i < rangeSizeUnblocked; i++) {
          zeros[i] = 0.0;
        }
      }
      deviceResultGridTranspose.intializeTo(zeros, 1, 0, rangeSizeUnblocked);
      // clFinish(device->commandQueue);
      // end = std::chrono::system_clock::now();
      // std::chrono::duration<double> elapsed_seconds = end - start;
      // if (verbose) {
      //   std::cout << "init buffers multTranspose: " <<
      //   elapsed_seconds.count()
      //             << std::endl;
      // }

      size_t rangeSizeBlocked = (kernelEndGrid / transGridBlockingSize) -
                                (kernelStartGrid / transGridBlockingSize);

      if (rangeSizeBlocked > 0) {

        // assuming transferring whole dataset
        int deviceGridOffset = kernelStartGrid;
        if (!transferWholeGrid) {
          deviceGridOffset = 0;
        }

        opencl::apply_arguments(this->kernelMultTranspose,
                                *(this->deviceLevelTranspose.getBuffer()),
                                *(this->deviceIndexTranspose.getBuffer()),
                                *(this->deviceDataTranspose.getBuffer()),
                                *(this->deviceSourceTranspose.getBuffer()),
                                *(this->deviceResultGridTranspose.getBuffer()),
                                deviceGridOffset,
                                static_cast<int>(start_index_data),
                                static_cast<int>(end_index_data));

        cl_event clTiming;

        // clFinish(device->commandQueue);

        const size_t rangeSizeBlocked2D[2] = {rangeSizeBlocked, dataSplit};
        const size_t localSize2D[2] = {localSize, 1};
        err = clEnqueueNDRangeKernel(device->commandQueue, kernelMultTranspose,
                                     2, 0, rangeSizeBlocked2D, localSize2D, 0,
                                     nullptr, &clTiming);

        if (err != CL_SUCCESS) {
          std::stringstream errorString;
          errorString
              << "OCL Error: Failed to enqueue kernel command! Error code: "
              << err << std::endl;
          throw sgpp::base::operation_exception(errorString.str());
        }

        clFinish(device->commandQueue);

        deviceResultGridTranspose.readFromBuffer();

        clFinish(device->commandQueue);

        std::vector<real_type> &deviceResultGridTransposeHost =
            deviceResultGridTranspose.getHostPointer();
        for (size_t i = 0; i < rangeSizeUnblocked; i++) {
          //                    std::cout << "resultDevice[" << i << "] = " <<
          //                    deviceResultGridTransposeHost[i] << std::endl;
          //                    std::cout << "-> result[" << kernelStartGrid + i
          //                    << "]" << std::endl;
          result[kernelStartGrid + i] = deviceResultGridTransposeHost[i];
        }

        // determine kernel execution time
        cl_ulong startTime = 0;
        cl_ulong endTime = 0;

        err = clGetEventProfilingInfo(clTiming, CL_PROFILING_COMMAND_START,
                                      sizeof(cl_ulong), &startTime, nullptr);

        if (err != CL_SUCCESS) {
          std::stringstream errorString;
          errorString << "OCL Error: Failed to read start-time from command "
                         "queue (or crash in multTranspose)! Error code: "
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

        double time = 0.0;
        time = static_cast<double>(endTime - startTime);
        time *= 1e-9;

        if (verbose) {
          std::cout << "device: " << device->deviceId << " duration: " << time
                    << std::endl;
        }

        this->deviceTimingMultTranspose += time;

        clReleaseEvent(clTiming);
      }
    }
    return this->deviceTimingMultTranspose;
  }
}; // namespace StreamingModOCLUnified

} // namespace StreamingModOCLUnified
} // namespace datadriven
} // namespace sgpp
