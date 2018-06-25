// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org
#pragma once
#include <sgpp/base/exception/operation_exception.hpp>
#include <sgpp/base/opencl/KernelSourceBuilderBase.hpp>
#include <sgpp/base/opencl/OCLOperationConfiguration.hpp>

#include <fstream>
#include <sstream>
#include <string>
namespace sgpp {
namespace datadriven {
namespace DensityOCLMultiPlatform {

/// OpenCL source builder for density right hand side vector
template <typename real_type>
class SourceBuilderB : public base::KernelSourceBuilderBase<real_type> {
 private:
  /// OpenCL configuration containing the building flags
  json::Node &kernelConfiguration;
  /// Dimensions of grid
  size_t dims;
  /// Used workgroupsize for opencl kernel execution
  size_t localWorkgroupSize;
  /// Using local memory?
  // bool useLocalMemory;
  // size_t dataBlockSize;
  // size_t transGridBlockSize;
  // uint64_t maxDimUnroll;

  // std::string getData(std::string dim, size_t dataBlockingIndex) {
  //   std::stringstream output;
  //   if (kernelConfiguration["KERNEL_STORE_DATA"].get().compare("array") == 0) {
  //     output << "data_" << dataBlockingIndex << "[" << dim << "]";
  //   } else if (kernelConfiguration["KERNEL_STORE_DATA"].get().compare("register") == 0) {
  //     output << "data_" << dataBlockingIndex << "_" << dim;
  //   } else if (kernelConfiguration["KERNEL_STORE_DATA"].get().compare("pointer") == 0) {
  //     output << "ptrData[(" << dataBlockSize << " * globalIdx) + (resultSize * " << dim << ") + "
  //            << dataBlockingIndex << "]";
  //   } else {
  //     std::string error("OCL Error: Illegal value for parameter \"KERNEL_STORE_DATA\"");
  //     throw new base::operation_exception(error.c_str());
  //   }
  //   return output.str();
  // }

 public:
  SourceBuilderB(json::Node &kernelConfiguration, size_t dims)
      : kernelConfiguration(kernelConfiguration), dims(dims) {}

  /// Generates the opencl source code for the density right hand side vector
    std::string generateSource(size_t dimensions, size_t data_points, size_t grid_points) {
    if (kernelConfiguration.contains("REUSE_SOURCE")) {
      if (kernelConfiguration["REUSE_SOURCE"].getBool()) {
        return this->reuseSource("DensityOCLMultiPlatform_rhs.cl");
      }
    }

    uint64_t local_size = kernelConfiguration["LOCAL_SIZE"].getUInt();
    
    std::stringstream sourceStream;

    if (this->floatType().compare("double") == 0) {
      sourceStream << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" << std::endl << std::endl;
    }

    sourceStream << "void kernel __attribute__((reqd_work_group_size(" << local_size << ", 1, 1))) cscheme(global const int* starting_points," << std::endl
                 << "global const " << this->floatType() << "* data_points,global "
                 << this->floatType() << "* C, private int startid) {" << std::endl;
    if (!kernelConfiguration["KERNEL_USE_LOCAL_MEMORY"].getBool()) {
      sourceStream << this->indent[0] << "C[get_global_id(0)]=0.0;" << std::endl
                   << this->indent[0] << "private " << this->floatType() << " value=1;" << std::endl
                   << this->indent[0] << "private " << this->floatType() << " wert=1.0;"
                   << std::endl
                   << this->indent[0] << "for(unsigned int ds=0;ds< " << data_points << ";ds++)"
                   << std::endl
                   << this->indent[0] << "{" << std::endl
                   << this->indent[1] << "value=1;" << std::endl
                   << this->indent[1] << "for(private int d=0;d< " << dimensions << ";d++)"
                   << std::endl
                   << this->indent[1] << "{" << std::endl
                   << this->indent[2] << "wert = (1 << starting_points[(startid + "
                   << " get_global_id(0))*2* " << dimensions << "+2*d+1]);" << std::endl
                   << this->indent[2] << "wert*=data_points[ds* " << dimensions << "+d];"
                   << std::endl
                   << this->indent[2] << "wert-=starting_points[(startid + get_global_id(0))*2* "
                   << dimensions << "+2*d];" << std::endl
                   << this->indent[3] << "wert=fabs(wert);" << std::endl
                   << this->indent[2] << "wert=1-wert;" << std::endl
                   << this->indent[2] << "if(wert<0)" << std::endl
                   << this->indent[3] << "wert=0;" << std::endl
                   << this->indent[2] << "value*=wert;" << std::endl
                   << this->indent[1] << "}" << std::endl
                   << this->indent[1] << "C[get_global_id(0)]+=value;" << std::endl
                   << this->indent[0] << "}" << std::endl
                   << this->indent[1] << "C[get_global_id(0)]/=" << data_points << ";" << std::endl
                   << "}" << std::endl;
    } else {
      // with local memory
  // int dims = 10;
  // int data_points_num = 16;

        sourceStream << this->indent[0] << this->floatType() << " grid_index[" << dimensions << "];"  << std::endl;
        sourceStream << this->indent[0] << this->floatType() << " grid_level_2[" << dimensions << "];" << std::endl;
        sourceStream << this->indent[0] <<  "if (get_global_id(0) < " << grid_points << ") {" << std::endl;
        sourceStream << this->indent[1] << "for (int d = 0; d < " << dimensions << "; d++) {" << std::endl;
        sourceStream << this->indent[2] << "grid_index[d] = (" << this->floatType() << ")(starting_points[(startid + get_global_id(0)) * 2 * " << dimensions << " + 2 * d]);" << std::endl;
        sourceStream << this->indent[2] << "grid_level_2[d] =" << std::endl;
        sourceStream << this->indent[3] << "(" << this->floatType() << ")(1 << starting_points[(startid + get_global_id(0)) * 2 * " << dimensions << " + 2 * d + 1]);" << std::endl;
        sourceStream << this->indent[1] << "}" << std::endl << std::endl;
        sourceStream << this->indent[0] <<  "} else {" << std::endl;
        sourceStream << this->indent[1] <<  "for (int d = 0; d < 10; d++) {" << std::endl;
        sourceStream << this->indent[2] <<  "grid_index[d] = 1.0;" << std::endl;
        sourceStream << this->indent[2] <<  "grid_level_2[d] = 2.0;" << std::endl;
        sourceStream << this->indent[1] <<  "}" << std::endl;
        sourceStream << this->indent[0] <<  "}" << std::endl;

        sourceStream << this->indent[0] << "local " << this->floatType() << " data_group[" << local_size << " * " << dimensions << "];" << std::endl << std::endl;

        sourceStream << this->indent[0] << this->floatType() << " result = 0.0" << this->constSuffix() <<";" << std::endl;
        sourceStream << this->indent[0] << "for (int outer_data_index = 0; outer_data_index < " << data_points << ";" << std::endl;
        sourceStream << this->indent[2] << "outer_data_index += get_local_size(0)) {" << std::endl;
        sourceStream << this->indent[1] << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
        sourceStream << this->indent[1] << "for (int d = 0; d < " << dimensions << "; d++) {" << std::endl;
        sourceStream << this->indent[2] << "int data_index = outer_data_index + get_local_id(0);" << std::endl;
        sourceStream << this->indent[2] << "if (data_index < " << data_points << ") {" << std::endl;
        sourceStream << this->indent[3] << "data_group[get_local_id(0) * " << dimensions << " + d] = data_points[data_index * " << dimensions << " + d];" << std::endl;
        sourceStream << this->indent[2] << "}" << std::endl;
        sourceStream << this->indent[1] << "}" << std::endl;
        sourceStream << this->indent[1] << "barrier(CLK_LOCAL_MEM_FENCE);"<< std::endl << std::endl;

        sourceStream << this->indent[1] << "for (int inner_data_index = 0; inner_data_index < get_local_size(0); inner_data_index += 1) {" << std::endl;
        sourceStream << this->indent[2] << "int data_index = outer_data_index + inner_data_index;" << std::endl;
        sourceStream << this->indent[2] << "if (data_index >= " << data_points << ") {" << std::endl;
        sourceStream << this->indent[3] << "break;" << std::endl;
        sourceStream << this->indent[2] << "}" << std::endl;
        sourceStream << this->indent[2] << this->floatType() << " eval = 1.0" << this->constSuffix() <<";" << std::endl;
        sourceStream << this->indent[2] << "for (int d = 0; d < " << dimensions << "; d++) {" << std::endl;
        sourceStream << this->indent[3] << this->floatType() << " eval_1d = grid_level_2[d];" << std::endl;
        sourceStream << this->indent[3] << "eval_1d *= data_group[inner_data_index * " << dimensions << " + d];" << std::endl;
        sourceStream << this->indent[3] << "eval_1d -= grid_index[d];" << std::endl;
        sourceStream << this->indent[3] << "eval_1d = fabs(eval_1d);" << std::endl;
        sourceStream << this->indent[3] << "eval_1d = 1 - eval_1d;" << std::endl;
        sourceStream << this->indent[3] << "if (eval_1d < 0) eval_1d = 0;" << std::endl;
        sourceStream << this->indent[3] << "eval *= eval_1d;" << std::endl;
        sourceStream << this->indent[2] << "}" << std::endl;
        sourceStream << this->indent[2] << "result += eval;" << std::endl;
        sourceStream << this->indent[1] << "}" << std::endl;
        sourceStream << this->indent[0] << "}" << std::endl;
        sourceStream << this->indent[0] << "result /= " << data_points << ".0" << this->constSuffix() << ";" << std::endl;
        sourceStream << this->indent[0] << "if (get_global_id(0) < " << grid_points << ") {" << std::endl;
        sourceStream << this->indent[1] << "C[get_global_id(0)] = result;" << std::endl;
        sourceStream << this->indent[0] << "}" << std::endl;
        sourceStream << "}" << std::endl;
    }
    if (kernelConfiguration["WRITE_SOURCE"].getBool() &&
        !kernelConfiguration["REUSE_SOURCE"].getBool()) {
      this->writeSource("DensityOCLMultiPlatform_rhs.cl", sourceStream.str());
    }
    return sourceStream.str();
  }
};

}  // namespace DensityOCLMultiPlatform
}  // namespace datadriven
}  // namespace sgpp
