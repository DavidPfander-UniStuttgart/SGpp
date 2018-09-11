// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org
#pragma once

#include <sgpp/base/exception/operation_exception.hpp>
#include <sgpp/base/grid/Grid.hpp>
#include <sgpp/base/grid/GridStorage.hpp>
#include <sgpp/base/opencl/OCLManager.hpp>
#include <sgpp/base/opencl/OCLOperationConfiguration.hpp>
#include <sgpp/base/operation/hash/OperationMatrix.hpp>
#include <sgpp/base/tools/SGppStopwatch.hpp>
#include <sgpp/globaldef.hpp>

#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

#include "KernelDensityB.hpp"
#include "KernelDensityMult.hpp"
#include "OperationDensity.hpp"

namespace sgpp {
namespace datadriven {
namespace DensityOCLMultiPlatform {

/// Class for opencl density multiplication and density right hand side vector
template <typename T>
class OperationDensityOCLMultiPlatform : public OperationDensity {
 private:
  size_t dims;
  size_t gridSize;
  /// OpenCL kernel which executes the matrix-vector density multiplications
  std::unique_ptr<sgpp::datadriven::DensityOCLMultiPlatform::KernelDensityMultInterface<T>> multKernel;
  /// OpenCL kernel which generates the right hand side vector of the density equation
  std::unique_ptr<sgpp::datadriven::DensityOCLMultiPlatform::KernelDensityBInterface<T>> bKernel;
  /// Vector with all OpenCL devices
  std::vector<std::shared_ptr<base::OCLDevice>> devices;
  /// Verbosity
  bool verbose;
  /// Contains all levels and indices of the sparse grid
  std::vector<int> points;
  /// OpenCL Manager
  std::shared_ptr<base::OCLManagerMultiPlatform> manager;
  /// Lambda for the density multiplication
  T lambda;

 public:
  /// Normal constructor
  OperationDensityOCLMultiPlatform(
      base::Grid &grid, size_t dimensions, std::shared_ptr<base::OCLManagerMultiPlatform> manager,
      std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters, T lambda)
      : OperationDensity(),
        dims(dimensions),
        gridSize(grid.getStorage().getSize()),
        devices(manager->getDevices()),
        verbose(false),
        manager(manager),
        lambda(lambda) {
    // Store Grid in a opencl compatible buffer
    sgpp::base::GridStorage &gridStorage = grid.getStorage();
    size_t pointscount = 0;
    for (size_t i = 0; i < gridSize; i++) {
      sgpp::base::HashGridPoint &point = gridStorage.getPoint(i);
      pointscount++;
      for (size_t d = 0; d < dims; d++) {
        points.push_back(point.getIndex(d));
        points.push_back(point.getLevel(d));
      }
    }
    for (size_t i = 0; i < devices.size(); i++) {
      std::cout << "device name: " << devices[i]->deviceName << std::endl;
    }

    if (devices.size() == 0) {
      std::stringstream errorString;
      errorString << "OperationDensityOCLMultiPlatform: No devices to use specified. Check you "
          "OpenCL configuration file"
                  << std::endl;
      throw base::operation_exception(errorString.str());
    } else if (devices.size() > 1) {
      std::stringstream errorString;
      errorString << "OperationDensityOCLMultiPlatform: need a single device to be specified, got "
                  << devices.size()
                  << " devices. Use the \"COUNT\" key in the configuration or remove sections "
          "of the configuration."
                  << std::endl;
      throw base::operation_exception(errorString.str());
    }
    auto device = devices[0];
    json::Node &deviceNode =
        (*parameters)["PLATFORMS"][device->platformName]["DEVICES"][device->deviceName];
    json::Node &firstKernelConfig = deviceNode["KERNELS"]["multdensity"];
    json::Node &secondKernelConfig = deviceNode["KERNELS"]["cscheme"];

    if (!secondKernelConfig.contains("USE_COMPRESSION_FIXED")) {
      bKernel =
          std::make_unique<KernelDensityB<T, uint64_t>>(device, dims, manager, secondKernelConfig, points);
    } else {
      if (!secondKernelConfig.contains("COMPRESSION_TYPE")) {
        bKernel =
            std::make_unique<KernelDensityB<T, uint64_t>>(device, dims, manager, secondKernelConfig, points);
      } else {
        if (secondKernelConfig["COMPRESSION_TYPE"].get().compare("uint64_t") == 0) {
          bKernel =
              std::make_unique<KernelDensityB<T, uint64_t>>(device, dims, manager, secondKernelConfig, points);
        } else if (secondKernelConfig["COMPRESSION_TYPE"].get().compare("unsigned int") == 0) {
          bKernel =
              std::make_unique<KernelDensityB<T, unsigned int>>(device, dims, manager, secondKernelConfig, points);
        } else {
          throw base::operation_exception(
              "OCL error: Illegal value for parameter \"COMPRESSION_TYPE\"\n");
        }
      }
    }

    if (!firstKernelConfig.contains("USE_COMPRESSION_FIXED")) {
      multKernel = std::make_unique<KernelDensityMult<T, uint64_t>>(device, dims, manager, firstKernelConfig,
                                                                    points, lambda);
    } else {
      if (!firstKernelConfig.contains("COMPRESSION_TYPE")) {
        multKernel = std::make_unique<KernelDensityMult<T, uint64_t>>(device, dims, manager, firstKernelConfig,
                                                                      points, lambda);
      } else {
        if (firstKernelConfig["COMPRESSION_TYPE"].get().compare("uint64_t") == 0) {
          multKernel = std::make_unique<KernelDensityMult<T, uint64_t>>(device, dims, manager, firstKernelConfig,
                                                                        points, lambda);
        } else if (firstKernelConfig["COMPRESSION_TYPE"].get().compare("unsigned int") == 0) {
          multKernel = std::make_unique<KernelDensityMult<T, unsigned int>>(device, dims, manager, firstKernelConfig,
                                                                            points, lambda);
        } else {
          throw base::operation_exception(
              "OCL error: Illegal value for parameter \"COMPRESSION_TYPE\"\n");
        }
      }
    }

    if (firstKernelConfig["VERBOSE"].getBool()) verbose = true;
  }
  /// Constructor for mpi nodes - accepts grid als integer array
  OperationDensityOCLMultiPlatform(
      const std::vector<int> &gridpoints, size_t gridsize, size_t dimensions,
      std::shared_ptr<base::OCLManagerMultiPlatform> manager,
      std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters, T lambda)
      : OperationDensity(),
        dims(dimensions),
        gridSize(gridsize),
        devices(manager->getDevices()),
        verbose(false),
        manager(manager),
        lambda(lambda) {
    for (size_t i = 0; i < gridSize; i++) {
      for (size_t d = 0; d < dims; d++) {
        points.push_back(gridpoints[2 * dimensions * i + 2 * d]);
        points.push_back(gridpoints[2 * dimensions * i + 2 * d + 1]);
      }
    }

    if (devices.size() == 0) {
      std::stringstream errorString;
      errorString << "OperationDensityOCLMultiPlatform: No devices to use specified. Check you "
          "OpenCL configuration file"
                  << std::endl;
      throw base::operation_exception(errorString.str());
    } else if (devices.size() > 1) {
      std::stringstream errorString;
      errorString << "OperationDensityOCLMultiPlatform: need a single device to be specified, got "
                  << devices.size()
                  << " devices. Use the \"COUNT\" key in the configuration or remove sections "
          "of the configuration."
                  << std::endl;
      throw base::operation_exception(errorString.str());
    }
    auto device = devices[0];
    json::Node &deviceNode =
        (*parameters)["PLATFORMS"][device->platformName]["DEVICES"][device->deviceName];
    json::Node &firstKernelConfig = deviceNode["KERNELS"]["multdensity"];
    json::Node &secondKernelConfig = deviceNode["KERNELS"]["cscheme"];


    if (!secondKernelConfig.contains("USE_COMPRESSION_FIXED")) {
      bKernel =
          std::make_unique<KernelDensityB<T, uint64_t>>(device, dims, manager, secondKernelConfig, points);
    } else {
      if (!secondKernelConfig.contains("COMPRESSION_TYPE")) {
        bKernel =
            std::make_unique<KernelDensityB<T, uint64_t>>(device, dims, manager, secondKernelConfig, points);
      } else {
        if (secondKernelConfig["COMPRESSION_TYPE"].get().compare("uint64_t") == 0) {
          bKernel =
              std::make_unique<KernelDensityB<T, uint64_t>>(device, dims, manager, secondKernelConfig, points);
        } else if (secondKernelConfig["COMPRESSION_TYPE"].get().compare("unsigned int") == 0) {
          bKernel =
              std::make_unique<KernelDensityB<T, unsigned int>>(device, dims, manager, secondKernelConfig, points);
        } else {
          throw base::operation_exception(
              "OCL error: Illegal value for parameter \"COMPRESSION_TYPE\"\n");
        }
      }
    }
    if (!firstKernelConfig.contains("USE_COMPRESSION_FIXED")) {
      multKernel = std::make_unique<KernelDensityMult<T, uint64_t>>(device, dims, manager, firstKernelConfig,
                                                                    points, lambda);
    } else {
      if (!firstKernelConfig.contains("COMPRESSION_TYPE")) {
        multKernel = std::make_unique<KernelDensityMult<T, uint64_t>>(device, dims, manager, firstKernelConfig,
                                                                      points, lambda);
      } else {
        if (firstKernelConfig["COMPRESSION_TYPE"].get().compare("uint64_t") == 0) {
          multKernel = std::make_unique<KernelDensityMult<T, uint64_t>>(device, dims, manager, firstKernelConfig,
                                                                        points, lambda);
        } else if (firstKernelConfig["COMPRESSION_TYPE"].get().compare("unsigned int") == 0) {
          multKernel = std::make_unique<KernelDensityMult<T, unsigned int>>(device, dims, manager, firstKernelConfig,
                                                                            points, lambda);
        } else {
          throw base::operation_exception(
              "OCL error: Illegal value for parameter \"COMPRESSION_TYPE\"\n");
        }
      }
    }
    if (firstKernelConfig["VERBOSE"].getBool()) verbose = true;
  }

  /// Use before calling partial_mult directly
  void initialize_alpha(std::vector<double> &alpha) override {
    std::vector<T> alphaVector(gridSize);
    for (size_t i = 0; i < gridSize; ++i) {
      alphaVector[i] = static_cast<T>(alpha[i]);
    }
    this->multKernel->initialize_alpha_buffer(alphaVector);
  }

  /// Execute a partial (startindex to startindex+chunksize) multiplication with the density matrix
  void start_partial_mult(size_t start_id, size_t chunksize) override {
    this->multKernel->start_mult(start_id, chunksize);
  }

  void finish_partial_mult(double *result, size_t start_id, size_t chunksize) override {
    if (std::is_same<T, double>::value) {
      std::vector<T> resultVector(result, result + chunksize);
      this->multKernel->finish_mult(resultVector, start_id, chunksize);
      std::copy(resultVector.begin(), resultVector.end(), result);
    } else {
      std::vector<T> resultVector(chunksize);
      for (int i = 0; i < chunksize; i++) {
        resultVector[i] = static_cast<T>(result[i]);
      }
      this->multKernel->finish_mult(resultVector, start_id, chunksize);
      std::copy(resultVector.begin(), resultVector.end(), result);
    }
  }

  /// Execute one matrix-vector multiplication with the density matrix
  void mult(base::DataVector &alpha, base::DataVector &result) override {
    std::vector<T> alphaVector(gridSize);
    std::vector<T> resultVector(gridSize);
    for (size_t i = 0; i < gridSize; i++) {
      alphaVector[i] = static_cast<T>(alpha[i]);
      resultVector[i] = static_cast<T>(result[i]);
    }
    if (verbose)
      std::cout << "starting multiplication with " << gridSize << " entries" << std::endl;
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    this->multKernel->initialize_alpha_buffer(alphaVector);
    this->multKernel->start_mult(0, 0);
    this->multKernel->finish_mult(resultVector, 0, 0);
    for (size_t i = 0; i < gridSize; i++) result[i] = resultVector[i];
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    this->last_duration_density = elapsed_seconds.count();
    this->acc_duration_density += this->last_duration_density;

    if (verbose) {
      std::cout << "duration mult ocl: " << elapsed_seconds.count() << std::endl;
    }
  }

  void initialize_dataset(base::DataMatrix &dataset) override {
    if (std::is_same<T, double>::value) {
      double *data_raw = dataset.getPointer();
      std::vector<T> datasetVector(data_raw, data_raw + dataset.getSize());
      bKernel->initialize_dataset(datasetVector);
    } else {
      std::vector<T> datasetVector(dataset.getSize());
      double *data_raw = dataset.getPointer();
      for (size_t i = 0; i < dataset.getSize(); i++) datasetVector[i] = static_cast<T>(data_raw[i]);
      bKernel->initialize_dataset(datasetVector);
    }
  }
  void start_rhs_generation(size_t start_id, size_t chunksize) override {
    bKernel->start_rhs_generation(start_id, chunksize);
  }

  void finalize_rhs_generation(sgpp::base::DataVector &b, size_t start_id,
                               size_t chunksize) override {
    std::vector<T> bVector(b.getSize());
    bKernel->finalize_rhs_generation(bVector, start_id, chunksize);
    for (size_t i = 0; i < b.getSize(); i++) b[i] = bVector[i];
  }

  /// Generates the right hand side vector for the density equation
  void generateb(base::DataMatrix &dataset, sgpp::base::DataVector &b, size_t start_id = 0,
                 size_t chunksize = 0) override {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    if (verbose) {
      if (chunksize == 0)
        std::cout << "starting rhs kernel methode! datasize: " << b.getSize() << std::endl;
      else
        std::cout << "starting rhs kernel methode! chunksize: " << chunksize << std::endl;
    }
    std::vector<T> bVector(b.getSize());
    std::vector<T> datasetVector(dataset.getSize());
    double *data_raw = dataset.getPointer();
    for (size_t i = 0; i < dataset.getSize(); i++) datasetVector[i] = static_cast<T>(data_raw[i]);
    start = std::chrono::system_clock::now();

    bKernel->initialize_dataset(datasetVector);
    bKernel->start_rhs_generation(start_id, chunksize);
    bKernel->finalize_rhs_generation(bVector, start_id, chunksize);

    for (size_t i = 0; i < b.getSize(); i++) {
      b[i] = bVector[i];
    }
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    this->last_duration_b = elapsed_seconds.count();
    this->acc_duration_b += this->last_duration_b;
    if (verbose) {
      std::cout << "duration generate b ocl: " << elapsed_seconds.count() << std::endl;
    }
  }
};

}  // namespace DensityOCLMultiPlatform
}  // namespace datadriven
}  // namespace sgpp
