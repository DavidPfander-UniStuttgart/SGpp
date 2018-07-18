// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#pragma once

#include <algorithm>
#include <fstream>
#include <string>

#include "sgpp/base/exception/operation_exception.hpp"
#include "sgpp/base/opencl/KernelSourceBuilderBase.hpp"
#include "sgpp/base/opencl/OCLOperationConfiguration.hpp"

namespace sgpp {
namespace datadriven {
namespace StreamingOCLMultiPlatform {

/**
 * Class for creating compute kernels for the transposed MultiEval operation \f$v':= B v\f$.
 * This class uses the parameters of the provided configuration to create OpenCL source code.
 * The generated code is specific to a single device, there each KernelMult requires its own
 * SourceBuilderMult.
 * For the code generation, a code fragment concatenation approach is used.
 *
 * @see StreamingOCLMultiPlatform::Configuration
 * @see StreamingOCLMultiPlatform::KernelMultTranspose
 */
template <typename real_type>
class SourceBuilderMultTranspose : public base::KernelSourceBuilderBase<real_type> {
 private:
  std::shared_ptr<base::OCLDevice> device;

  json::Node &kernelConfiguration;

  size_t dims;

  size_t localWorkgroupSize;
  bool useLocalMemory;
  bool use_compression;
  size_t transGridBlockSize;
  uint64_t maxDimUnroll;
  size_t transPrefetchSize;

  /**
   * Generates code fragments that models an access to the level variable.
   *
   * @param dim dimension to be accessed
   * @param gridBlockingIndex grid blocking information to be taken into account
   */
  std::string getLevel(std::string dim, size_t gridBlockingIndex) {
    std::stringstream output;
    if (kernelConfiguration["KERNEL_STORE_DATA"].get().compare("array") == 0) {
      output << "level_" << gridBlockingIndex << "[" << dim << "]";
    } else if (kernelConfiguration["KERNEL_STORE_DATA"].get().compare("register") == 0) {
      output << "level_" << gridBlockingIndex << "_" << dim;
    } else if (kernelConfiguration["KERNEL_STORE_DATA"].get().compare("pointer") == 0) {
      output << "ptrLevel[dimLevelIndex]";
    } else if (kernelConfiguration["KERNEL_STORE_DATA"].get().compare("compressed") == 0) {
      output << "decompressed_level";
    } else {
      throw base::operation_exception(
          "OCL error: Illegal value for parameter \"KERNEL_STORE_DATA\"\n");
    }
    return output.str();
  }

  /**
   * Generates code fragments that models an access to the index variable.
   *
   * @param dim dimension to be accessed
   * @param gridBlockingIndex grid blocking information to be taken into account
   */
  std::string getIndex(std::string dim, size_t gridBlockingIndex) {
    std::stringstream output;
    if (kernelConfiguration["KERNEL_STORE_DATA"].get().compare("array") == 0) {
      output << "index_" << gridBlockingIndex << "[" << dim << "]";
    } else if (kernelConfiguration["KERNEL_STORE_DATA"].get().compare("register") == 0) {
      output << "index_" << gridBlockingIndex << "_" << dim;
    } else if (kernelConfiguration["KERNEL_STORE_DATA"].get().compare("pointer") == 0) {
      output << "ptrIndex[dimLevelIndex]";
    } else if (kernelConfiguration["KERNEL_STORE_DATA"].get().compare("compressed") == 0) {
      output << "decompressed_index";
    } else {
      throw base::operation_exception(
          "OCL error: Illegal value for parameter \"KERNEL_STORE_DATA\"\n");
    }
    return output.str();
  }

  /**
   * Generates code fragments that models a dataset access.
   *
   * @param dim dimension to be accessed
   * @param dataBlockingIndex data blocking information to be taken into account
   */
  std::string getData(std::string dim, size_t dataBlockingIndex) {
    std::stringstream output;
    if (kernelConfiguration["KERNEL_USE_LOCAL_MEMORY"].getBool()) {
      output << "locData[(" << dim << " * " << transPrefetchSize << ") + k]";
    } else {
      output << "ptrData[(" << dim << " * sourceSize) + k]";
    }
    return output.str();
  }

  /**
   * Generates code fragments for an basis function evaluation.
   * Creates only dimension after another without a loop.
   *
   * @param dims Number of dimensions to be implemented
   * @param startDim First dimension of the unroll, useful for implementing loop unrolling
   * @param endDim Last dimension of the unrolled implementation
   * @param unrollVariable Variable used for partial unrolling, can be an empty string if everything
   * is unrolled
   */
  std::string unrolledBasisFunctionEvalulation(size_t dims, size_t startDim, size_t endDim,
                                               std::string unrollVariable) {
    std::stringstream output;

    for (size_t d = startDim; d < endDim; d++) {
      std::stringstream dimElement;
      dimElement << "(";
      if (!unrollVariable.compare("") == 0) {
        dimElement << unrollVariable << " + ";
      }
      dimElement << d;
      dimElement << ")";
      std::string pointerAccess = dimElement.str();

      std::string dString;
      if (kernelConfiguration["KERNEL_STORE_DATA"].get().compare("register") == 0) {
        std::stringstream stream;
        stream << (d);
        dString = stream.str();
      } else {
        dString = pointerAccess;
      }


      for (size_t gridPoint = 0; gridPoint < transGridBlockSize; gridPoint++) {
      if (use_compression) {
        output << this->indent[3]
               << "ulong is_dim_implicit = fixed_dim_zero_flags & one_mask;" << std::endl;
        output << this->indent[3] << "fixed_dim_zero_flags >>= 1;" << std::endl;
        output << this->indent[3] << "ulong decompressed_level = 1;" << std::endl;
        output << this->indent[3] << "ulong decompressed_index = 1;" << std::endl;
        output << this->indent[3] << "if (is_dim_implicit != 0) {" << std::endl;
        output << this->indent[4] << "ulong level_bits = 1 + "
               << "clz(fixed_level_offsets);"
               << std::endl;
        output << this->indent[4] << "fixed_level_offsets <<= level_bits;" << std::endl;
        output << this->indent[4] << "ulong level_mask = (1 << level_bits) - 1;" << std::endl;
        output << this->indent[4] << "decompressed_level = (fixed_level_packed & level_mask) + 2;" << std::endl;
        output << this->indent[4] << "fixed_level_packed >>= level_bits;" << std::endl;
        output << this->indent[4] << "ulong index_bits = decompressed_level - 1;" << std::endl;
        output << this->indent[4] << "ulong index_mask = (1 << index_bits) - 1;" << std::endl;
        output << this->indent[4] << "decompressed_index = ((fixed_index_packed & index_mask) << 1) + 1;" << std::endl;
        output << this->indent[4] << "fixed_index_packed >>= index_bits;" << std::endl;
        output << this->indent[3] << "}" << std::endl;
      }


        output << this->indent[2] << "curSupport_" << gridPoint << " *= fmax(1.0"
               << this->constSuffix() << " - fabs((";
        output << getLevel(dString, gridPoint) << " * " << getData(dString, 0) << ") - "
               << getIndex(dString, gridPoint) << "), 0.0" << this->constSuffix() << ");"
               << std::endl;
      }
    }
    return output.str();
  }

 public:
  /**
   * Constructor
   *
   * @param device The device this code generators creates for
   * @param kernelConfiguration The configuration for this device
   * @param dims Dimension of the data mining problem
   */
  SourceBuilderMultTranspose(std::shared_ptr<base::OCLDevice> device,
                             json::Node &kernelConfiguration, size_t dims)
      : device(device), kernelConfiguration(kernelConfiguration), dims(dims) {
    localWorkgroupSize = kernelConfiguration["LOCAL_SIZE"].getUInt();
    useLocalMemory = kernelConfiguration["KERNEL_USE_LOCAL_MEMORY"].getBool();
    transGridBlockSize = kernelConfiguration["KERNEL_TRANS_GRID_BLOCK_SIZE"].getUInt();
    maxDimUnroll = kernelConfiguration["KERNEL_MAX_DIM_UNROLL"].getUInt();
    transPrefetchSize = kernelConfiguration["KERNEL_TRANS_PREFETCH_SIZE"].getUInt();
    if (kernelConfiguration["KERNEL_STORE_DATA"].get().compare("compressed") == 0) {
      use_compression = true;
    } else {
      use_compression = false;
    }
  }

  /**
   * Entry point of source generator.
   * Creates compute kernel for the transposed MultiEval operation.
   */
  std::string generateSource() {
    if (kernelConfiguration["REUSE_SOURCE"].getBool()) {
      return this->reuseSource("StreamingOCLMultiPlatform_multTrans.cl");
    }

    std::stringstream sourceStream;

    if (std::is_same<real_type, double>::value) {
      sourceStream << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" << std::endl << std::endl;
    }

    sourceStream << "__kernel" << std::endl;

    sourceStream << "__attribute__((reqd_work_group_size(" << localWorkgroupSize << ", 1, 1)))"
                 << std::endl;
    sourceStream << "void multTransOCL(";
    if (!use_compression) {
      sourceStream << "__global const " << this->floatType() << "* ptrLevel," << std::endl;
      sourceStream << "                  __global const " << this->floatType() << "* ptrIndex,"
                   << std::endl;
      sourceStream << "                  __global const " << this->floatType() << "* ptrData,"
                   << std::endl;
    } else {
      sourceStream << "__global const ulong *dim_zero_flags_v, "
                   << "__global const ulong *level_offsets_v, "
                   << "__global const ulong *level_packed_v, "
                   << "__global const ulong *index_packed_v, ";
    }
    sourceStream << "                  __global const " << this->floatType() << "* ptrSource,"
                 << std::endl;
    sourceStream << "                  __global       " << this->floatType() << "* ptrResult,"
                 << std::endl;
    sourceStream << "                  int sourceSize," << std::endl;
    sourceStream << "                  int start_data," << std::endl;
    sourceStream << "                  int end_data) {" << std::endl;
    sourceStream << this->indent[0] << "int globalIdx = get_global_id(0);" << std::endl;
    sourceStream << this->indent[0] << "int localIdx = get_local_id(0);" << std::endl;
    sourceStream << this->indent[0] << "int globalSize = get_global_size(0);" << std::endl;
    sourceStream << std::endl;

    for (size_t gridPoint = 0; gridPoint < transGridBlockSize; gridPoint++) {
      sourceStream << this->indent[0] << this->floatType() << " myResult_" << gridPoint << " = 0.0;"
                   << std::endl;
    }
    sourceStream << std::endl;

    if (useLocalMemory) {
      sourceStream << this->indent[0] << "__local " << this->floatType() << " locData["
                   << dims * transPrefetchSize << "];" << std::endl;
      sourceStream << this->indent[0] << "__local " << this->floatType() << " locSource["
                   << transPrefetchSize << "];" << std::endl
                   << std::endl;
    }

    // create a register storage for the level and index of the grid points of
    // the work item
    if (kernelConfiguration["KERNEL_STORE_DATA"].get().compare("array") == 0) {
      for (size_t gridPoint = 0; gridPoint < transGridBlockSize; gridPoint++) {
        sourceStream << this->indent[0] << this->floatType() << " level_" << gridPoint << "["
                     << dims << "];" << std::endl;
        for (size_t d = 0; d < dims; d++) {
          sourceStream << this->indent[0] << "level_" << gridPoint << "[" << d
                       << "] = ptrLevel[(((globalSize * " << gridPoint << ") + globalIdx) * "
                       << dims << ") + " << d << "];" << std::endl;
        }
        sourceStream << std::endl;

        sourceStream << this->indent[0] << this->floatType() << " index_" << gridPoint << "["
                     << dims << "];" << std::endl;
        for (size_t d = 0; d < dims; d++) {
          sourceStream << this->indent[0] << "index_" << gridPoint << "[" << d
                       << "] = ptrIndex[(((globalSize * " << gridPoint << ") + globalIdx) * "
                       << dims << ") + " << d << "];" << std::endl;
        }
        sourceStream << std::endl;
      }
    } else if (kernelConfiguration["KERNEL_STORE_DATA"].get().compare("register") == 0) {
      for (size_t gridPoint = 0; gridPoint < transGridBlockSize; gridPoint++) {
        for (size_t d = 0; d < dims; d++) {
          sourceStream << this->indent[0] << this->floatType() << " level_" << gridPoint << "_" << d
                       << " = ptrLevel[(((globalSize * " << gridPoint << ") + globalIdx) * " << dims
                       << ") + " << d << "];" << std::endl;
        }
        sourceStream << std::endl;

        for (size_t d = 0; d < dims; d++) {
          sourceStream << this->indent[0] << this->floatType() << " index_" << gridPoint << "_" << d
                       << " = ptrIndex[(((globalSize * " << gridPoint << ") + globalIdx) * " << dims
                       << ") + " << d << "];" << std::endl;
        }
        sourceStream << std::endl;
      }
    } else if (kernelConfiguration["KERNEL_STORE_DATA"].get().compare("compressed") == 0) {
      for (size_t gridPoint = 0; gridPoint < transGridBlockSize; gridPoint++) {
        sourceStream << this->indent[0] << " ulong one_mask = 1;" << std::endl;
        sourceStream << this->indent[0] << "__private ulong point_dim_zero_flags = "
                     << " dim_zero_flags_v[((globalSize *" << gridPoint << ") + globalIdx)];"
                     << std::endl;
        sourceStream << this->indent[0] << "__private ulong point_level_offsets = "
                     << " level_offsets_v[((globalSize *" << gridPoint << ") + globalIdx)];"
                     << std::endl;
        sourceStream << this->indent[0] << "__private ulong point_level_packed = "
                     << " level_packed_v[((globalSize *" << gridPoint << ") + globalIdx)];"
                     << std::endl;
        sourceStream << this->indent[0] << "__private ulong point_index_packed = "
                     << " index_packed_v[((globalSize *" << gridPoint << ") + globalIdx)];"
                     << std::endl;
      }
    }

    sourceStream << this->indent[0] << "// Iterate over all data points" << std::endl;

    if (useLocalMemory) {
      sourceStream << this->indent[0] << "for(int dataBlockStart = start_data; dataBlockStart < "
          "end_data; dataBlockStart += "
                   << transPrefetchSize << ") {" << std::endl;

      sourceStream << this->indent[1] << "if (localIdx < " << transPrefetchSize << ") {"
                   << std::endl;
      if (dims > maxDimUnroll) {
        sourceStream << this->indent[1] << "for (int d = 0; d < " << dims << "; d++) {"
                     << std::endl;
        sourceStream << this->indent[2] << "locData[(" << transPrefetchSize
                     << " * d) + localIdx] = ptrData[(d * sourceSize) + "
            "dataBlockStart + localIdx];"
                     << std::endl;
        sourceStream << this->indent[1] << "}" << std::endl;
      } else {
        for (size_t d = 0; d < dims; d++) {
          sourceStream << this->indent[1] << "locData[(" << transPrefetchSize << " * " << d
                       << ") + localIdx] = ptrData[(" << d
                       << " * sourceSize) + dataBlockStart + localIdx];" << std::endl;
        }
      }

      sourceStream << this->indent[1]
                   << "locSource[localIdx] = ptrSource[localIdx + dataBlockStart];" << std::endl;

      sourceStream << this->indent[1] << "}" << std::endl;
      sourceStream << this->indent[1] << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
      sourceStream << std::endl;
      sourceStream << this->indent[1] << "for(int k = 0; k < " << transPrefetchSize << "; k++) {"
                   << std::endl;

      for (size_t gridPoint = 0; gridPoint < transGridBlockSize; gridPoint++) {
        sourceStream << this->indent[2] << this->floatType() << " curSupport_" << gridPoint
                     << " = locSource[k];" << std::endl;
      }
    } else {
      sourceStream << this->indent[0] << "for(int k = start_data; k < end_data; k++) {"
                   << std::endl;

      for (size_t gridPoint = 0; gridPoint < transGridBlockSize; gridPoint++) {
        sourceStream << this->indent[1] << this->floatType() << " curSupport_" << gridPoint
                     << " = ptrSource[k];" << std::endl;
      }
    }
    sourceStream << std::endl;

    if (dims > maxDimUnroll) {
      sourceStream << this->indent[1] << "for (int unrollDim = 0; unrollDim < "
                   << ((dims / maxDimUnroll) * maxDimUnroll) << "; unrollDim += " << maxDimUnroll
                   << ") {" << std::endl;

      sourceStream << this->unrolledBasisFunctionEvalulation(dims, 0, std::min(maxDimUnroll, dims),
                                                             "unrollDim");
      sourceStream << this->indent[1] << "}" << std::endl;

      if (dims % maxDimUnroll != 0) {
        sourceStream << this->unrolledBasisFunctionEvalulation(
            dims, (dims / maxDimUnroll) * maxDimUnroll, dims, "");
      }

    } else {
      sourceStream << this->unrolledBasisFunctionEvalulation(dims, 0, dims, "");
    }

    sourceStream << std::endl;

    for (size_t gridPoint = 0; gridPoint < transGridBlockSize; gridPoint++) {
      sourceStream << this->indent[1] << "myResult_" << gridPoint << " += curSupport_" << gridPoint
                   << ";" << std::endl;
    }
    sourceStream << this->indent[0] << "}" << std::endl << std::endl;

    if (useLocalMemory) {
      sourceStream << this->indent[1] << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
      sourceStream << this->indent[0] << "}" << std::endl;
    }

    for (size_t gridPoint = 0; gridPoint < transGridBlockSize; gridPoint++) {
      sourceStream << this->indent[0] << "ptrResult[(globalSize * " << gridPoint
                   << ") + globalIdx] = myResult_" << gridPoint << ";" << std::endl;
    }
    sourceStream << "}" << std::endl;

    if (kernelConfiguration["WRITE_SOURCE"].getBool()) {
      this->writeSource("StreamingOCLMultiPlatform_multTrans.cl", sourceStream.str());
    }

    return sourceStream.str();
  }
};
}  // namespace StreamingOCLMultiPlatform
}  // namespace datadriven
}  // namespace sgpp
