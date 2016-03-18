// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org
#pragma once

#include <sgpp/base/exception/operation_exception.hpp>
#include <sgpp/base/opencl/OCLOperationConfiguration.hpp>
#include <sgpp/base/opencl/KernelSourceBuilderBase.hpp>

#include <fstream>
#include <sstream>
#include <string>
namespace sgpp {
namespace datadriven {
namespace DensityOCLMultiPlatform {

template<typename real_type>
class SourceBuilderCreateGraph: public base::KernelSourceBuilderBase<real_type> {
 private:
  std::shared_ptr<base::OCLDevice> device;

  json::Node &kernelConfiguration;

  size_t dims;

  size_t localWorkgroupSize;
  bool useLocalMemory;
  size_t dataBlockSize;
  size_t transGridBlockSize;
  uint64_t maxDimUnroll;

  std::string getData(std::string dim, size_t dataBlockingIndex) {
    std::stringstream output;
    if (kernelConfiguration["KERNEL_STORE_DATA"].get().compare("array") == 0) {
      output << "data_" << dataBlockingIndex << "[" << dim << "]";
    } else if (kernelConfiguration["KERNEL_STORE_DATA"].get().compare("register") == 0) {
      output << "data_" << dataBlockingIndex << "_" << dim;
    } else if (kernelConfiguration["KERNEL_STORE_DATA"].get().compare("pointer") == 0) {
      output << "ptrData[(" << dataBlockSize << " * globalIdx) + (resultSize * " << dim
             << ") + " << dataBlockingIndex << "]";
    } else {
      std::string error("OCL Error: Illegal value for parameter \"KERNEL_STORE_DATA\"");
      throw new base::operation_exception(error.c_str());
    }
    return output.str();
  }

  std::string init_k_registers(size_t k) {
    std::stringstream output;
    output << this->indent[0] << "int k_reg["<< k << "];" << std::endl;
    output << this->indent[0] << this->floatType() << " k_dists["<< k << "];" << std::endl;
    output << this->indent[0] << "for (int i = 0; i < " << k << "; i++)" << std::endl;
    output << this->indent[1] << "k_dists[i] = 4.0;" << std::endl;
    /*for (size_t i = 0; i < k; i++) {
      output << this->indent[0] << "int k_register" << i << " = " << i << "; " << std::endl;
      }
      for (size_t i = 0; i < k; i++) {
      output << this->indent[0] << this->floatType()
      << " dist_k" << i << " = 4.0;" << std::endl;
      }*/
    return output.str();
  }
  std::string find_max_index(size_t k, bool unroll) {
    std::stringstream output;
    if (!unroll) {
      output << this->indent[2] << "for (unsigned int j = 1; j < "
             << k << "; j++) {" << std::endl
             << this->indent[3] << "if (k_dists[maxindex] < k_dists[j])" << std::endl
             << this->indent[4] << "maxindex = j;" << std::endl
             << this->indent[2] << "}" << std::endl;
    } else {
    for (size_t i = 1; i < k; i++) {
      output << this->indent[2] << "if (k_dists[maxindex] < k_dists[" << i << "])" << std::endl;
      output << this->indent[3] << "maxindex  = " << i << "; " << std::endl;
      }
    }
    // Enables vectorization but slows kernel down (longer ifs...)
    /*for (size_t i = 0; i < k; i++) {
      output << this->indent[2] << "if (maxdist < k_dists[" << i << "]) {" << std::endl;
      output << this->indent[3] << "maxindex  = " << i << "; " << std::endl;
      output << this->indent[3] << "maxdist  = k_dists["  << i << "]; }" << std::endl;
      }*/
    return output.str();
  }
  std::string save_from_global_to_local(size_t dimensions) {
    std::stringstream output;
    output << this->indent[0] << this->floatType() << " datapoint["
           << dimensions << "];" << std::endl;
    for (size_t i = 0; i < dimensions; i++) {
      output << this->indent[1] << "datapoint[" << i << "] = data[global_index * "
             << dimensions << " + " << i << "];" << std::endl;
    }
    return output.str();
  }
  std::string copy_k_registers_into_global(size_t k) {
    std::stringstream output;
    for (size_t i = 0; i < k; i++) {
      output << this->indent[0] << "neighbors[chunk_index * "<< k
             << " + " << i << "] = k_reg[" << i << "];" << std::endl;
    }
    return output.str();
  }

 public:
  SourceBuilderCreateGraph(std::shared_ptr<base::OCLDevice> device,
                           json::Node &kernelConfiguration, size_t dims) :
      device(device), kernelConfiguration(kernelConfiguration), dims(dims) {
  }

  std::string generateSource(size_t dimensions, size_t k, size_t dataSize) {
    if (kernelConfiguration.contains("REUSE_SOURCE")) {
      if (kernelConfiguration["REUSE_SOURCE"].getBool()) {
        return this->reuseSource("DensityOCLMultiPlatform_create_graph.cl");
      }
    }

    std::stringstream sourceStream;

    if (this->floatType().compare("double") == 0) {
      sourceStream << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable"
                   << std::endl << std::endl;
    }

    sourceStream << "" << std::endl
                 << "void kernel connectNeighbors(global " << this->floatType()
                 << " *data, global int *neighbors, int startid)" << std::endl
                 << "{" << std::endl
                 << this->indent[0] << "size_t global_index = startid + get_global_id(0);"
                 << std::endl
                 << this->indent[0] << "size_t chunk_index = get_global_id(0);" << std::endl
                 << init_k_registers(k)
                 << save_from_global_to_local(dimensions)
                 << this->indent[2] << this->floatType() << " dist = 0.0;" << std::endl
                 << this->indent[0] << "for (unsigned int i = 0; i <    " << dataSize
                 << "; i++) {" << std::endl
                 << this->indent[1] << "if (i != global_index) {" << std::endl
                 << "//get distance to current point" << std::endl
                 << this->indent[2] << "dist = 0.0;" << std::endl
                 << this->indent[2] << "for (unsigned int j = 0; j <     " << dimensions
                 << " ; j++) {" << std::endl
                 << this->indent[3] << "dist += (datapoint[j] - data[j + i* "
                 << dimensions << " ])" << std::endl
                 << this->indent[3] << "* (datapoint[j] - data[j + i* "
                 << dimensions << " ]);" << std::endl
                 << this->indent[2] << "}" << std::endl
                 << this->indent[2] << "int maxindex = 0;" << std::endl
                 << find_max_index(k, true)
                 << this->indent[2] << "if (dist < k_dists[maxindex]) {" << std::endl
                 << this->indent[3] << "k_reg[maxindex] = i;" << std::endl
                 << this->indent[3] << "k_dists[maxindex] = dist;" << std::endl
                 << this->indent[2] << "}" << std::endl
                 << this->indent[1] << "}" << std::endl
                 << this->indent[0] << "}" << std::endl
                 << copy_k_registers_into_global(k)
                 << "}" << std::endl;
    if (kernelConfiguration.contains("WRITE_SOURCE")) {
      if (kernelConfiguration["WRITE_SOURCE"].getBool()) {
        this->writeSource("DensityOCLMultiPlatform_create_graph.cl", sourceStream.str());
      }
    }
    return sourceStream.str();
  }
};

}  // namespace DensityOCLMultiPlatform
}  // namespace datadriven
}  // namespace sgpp