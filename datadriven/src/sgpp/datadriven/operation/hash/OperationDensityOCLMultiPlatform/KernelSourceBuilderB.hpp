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
  bool use_compression_fixed;

 public:
  SourceBuilderB(json::Node &kernelConfiguration, size_t dims)
      : kernelConfiguration(kernelConfiguration), dims(dims) {
    if (kernelConfiguration.contains("USE_COMPRESSION_FIXED"))
      use_compression_fixed = kernelConfiguration["USE_COMPRESSION_FIXED"].getBool();
    else
      use_compression_fixed = false;
  }

  /// Generates the opencl source code for the density right hand side vector
  std::string generateSource(size_t dimensions, size_t data_points, size_t grid_points) {
    if (kernelConfiguration.contains("REUSE_SOURCE")) {
      if (kernelConfiguration["REUSE_SOURCE"].getBool()) {
        return this->reuseSource("DensityOCLMultiPlatform_rhs.cl");
      }
    }

    uint64_t local_size = kernelConfiguration["LOCAL_SIZE"].getUInt();
    uint64_t local_cache_size = kernelConfiguration["KERNEL_LOCAL_CACHE_SIZE"].getUInt();

    std::stringstream sourceStream;

    if (this->floatType().compare("double") == 0) {
      sourceStream << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" << std::endl << std::endl;
    }

    // TODO: startid currently not used appropriately! -> add startid + get_global_id() to kernel
    if (!use_compression_fixed) {
      sourceStream << "void kernel __attribute__((reqd_work_group_size(" << local_size
                   << ", 1, 1))) cscheme(global const int* starting_points," << std::endl
                   << "global const " << this->floatType() << "* data_points,global "
                   << this->floatType() << "* C, private int startid) {" << std::endl;
    } else {
      sourceStream << "void kernel __attribute__((reqd_work_group_size(" << local_size
                   << ", 1, 1))) cscheme("
                   << "__global const ulong *dim_zero_flags_v, "
                   << "__global const ulong *level_offsets_v, "
                   << "__global const ulong *level_packed_v, "
                   << "__global const ulong *index_packed_v, "
                   << "global const " << this->floatType() << "* data_points,global "
                   << this->floatType() << "* C, private int startid) {" << std::endl;
      sourceStream << this->indent[0] << "int gridindex = startid + get_global_id(0);" << std::endl;
    }
    if (!kernelConfiguration["KERNEL_USE_LOCAL_MEMORY"].getBool()) {
      sourceStream << this->indent[0] << "C[get_global_id(0)]=0.0;" << std::endl
                   << this->indent[0] << "private " << this->floatType() << " value=1;" << std::endl
                   << this->indent[0] << "private " << this->floatType() << " wert=1.0;";
      if(use_compression_fixed) {
        sourceStream << this->indent[0] << " ulong one_mask = 1;" << std::endl;
        sourceStream << this->indent[0] << "__private ulong point_dim_zero_flags = 0;"
                     << std::endl;
        sourceStream << this->indent[0] << "__private ulong point_level_offsets = 0;"
                     << std::endl;
        sourceStream << this->indent[0] << "__private ulong point_level_packed = 0;"
                     << std::endl;
        sourceStream << this->indent[0] << "__private ulong point_index_packed = 0;"
                     << std::endl;

        sourceStream << this->indent[0] << "if (get_global_id(0) < " << grid_points << ") {"
                     << std::endl;
        sourceStream << this->indent[1] << "point_dim_zero_flags = dim_zero_flags_v[gridindex];"
                     << std::endl;
        sourceStream << this->indent[1] << "point_level_offsets = level_offsets_v[gridindex];"
                     << std::endl;
        sourceStream << this->indent[1] << "point_level_packed = level_packed_v[gridindex];"
                     << std::endl;
        sourceStream << this->indent[1] << "point_index_packed = index_packed_v[gridindex];"
                     << std::endl;
        sourceStream << this->indent[0] << "}" << std::endl;
      }
      sourceStream << std::endl
                   << this->indent[0] << "for(unsigned int ds=0;ds< " << data_points << ";ds++)"
                   << std::endl
                   << this->indent[0] << "{" << std::endl
                   << this->indent[1] << "value=1;" << std::endl;
      if (use_compression_fixed) {
        sourceStream << this->indent[2] << "ulong fixed_dim_zero_flags = point_dim_zero_flags;" << std::endl;
        sourceStream << this->indent[2] << "ulong fixed_level_offsets = point_level_offsets;" << std::endl;
        sourceStream << this->indent[2] << "ulong fixed_level_packed = point_level_packed;" << std::endl;
        sourceStream << this->indent[2] << "ulong fixed_index_packed = point_index_packed;" << std::endl;
      }
      sourceStream << this->indent[1] << "for(private int d=0;d< " << dimensions << ";d++)"
                   << std::endl
                   << this->indent[1] << "{" << std::endl;

      std::string index_func =
          std::string("starting_points[(startid + get_global_id(0))*2*") +
          std::to_string(dimensions) + std::string("+2*d]");
      std::string level_func =
          std::string("starting_points[(startid + get_global_id(0))*2*") +
          std::to_string(dimensions) + std::string("+2*d+1]");
      if (use_compression_fixed) {
        sourceStream << this->indent[3]
                     << "ulong is_dim_implicit = fixed_dim_zero_flags & one_mask;" << std::endl;
        sourceStream << this->indent[3] << "fixed_dim_zero_flags >>= 1;" << std::endl;
        sourceStream << this->indent[3] << "ulong decompressed_level = 1;" << std::endl;
        sourceStream << this->indent[3] << "ulong decompressed_index = 1;" << std::endl;
        sourceStream << this->indent[3] << "if (is_dim_implicit != 0) {" << std::endl;
        sourceStream << this->indent[4] << "ulong level_bits = 1 + "
                     << "clz(fixed_level_offsets);"
                     << std::endl;
        sourceStream << this->indent[4] << "fixed_level_offsets <<= level_bits;" << std::endl;
        sourceStream << this->indent[4] << "ulong level_mask = (1 << level_bits) - 1;" << std::endl;
        sourceStream << this->indent[4] << "decompressed_level = (fixed_level_packed & level_mask) + 2;" << std::endl;
        sourceStream << this->indent[4] << "fixed_level_packed >>= level_bits;" << std::endl;
        sourceStream << this->indent[4] << "ulong index_bits = decompressed_level - 1;" << std::endl;
        sourceStream << this->indent[4] << "ulong index_mask = (1 << index_bits) - 1;" << std::endl;
        sourceStream << this->indent[4] << "decompressed_index = ((fixed_index_packed & index_mask) << 1) + 1;" << std::endl;
        sourceStream << this->indent[4] << "fixed_index_packed >>= index_bits;" << std::endl;
        sourceStream << this->indent[3] << "}" << std::endl;
        level_func =
            std::string("decompressed_level");
        index_func =
            std::string("decompressed_index");
      }

      sourceStream << this->indent[2] << "wert = (1 << " << level_func << ");" << std::endl
                   << this->indent[2] << "wert*=data_points[ds* " << dimensions << "+d];"
                   << std::endl
                   << this->indent[2] << "wert-=" << index_func << ";" << std::endl
                   << this->indent[3] << "wert=fabs(wert);" << std::endl
                   << this->indent[2] << "wert=1-wert;" << std::endl
                   << this->indent[2] << "if(wert<0)" << std::endl
                   << this->indent[3] << "wert=0;" << std::endl
                   << this->indent[2] << "value*=wert;" << std::endl
                   << this->indent[1] << "}" << std::endl
                   << this->indent[1] << "C[get_global_id(0)]+=value;" << std::endl
                   << this->indent[0] << "}" << std::endl
                   << this->indent[1] << "C[get_global_id(0)]/=" << data_points << ";" << std::endl;
     sourceStream << "}" << std::endl;
    } else {
        // with local memory
        if (!use_compression_fixed) {
          sourceStream << this->indent[0] << this->floatType() << " grid_index[" << dimensions << "];"
                       << std::endl;
          sourceStream << this->indent[0] << this->floatType() << " grid_level_2[" << dimensions << "];"
                       << std::endl;
          sourceStream << this->indent[0] << "if (get_global_id(0) < " << grid_points << ") {"
                       << std::endl;
          sourceStream << this->indent[1] << "for (int d = 0; d < " << dimensions << "; d++) {"
                       << std::endl;
          sourceStream << this->indent[2] << "grid_index[d] = (" << this->floatType()
                       << ")(starting_points[(startid + get_global_id(0)) * 2 * " << dimensions
                       << " + 2 * d]);" << std::endl;
          sourceStream << this->indent[2] << "grid_level_2[d] =" << std::endl;
          sourceStream << this->indent[3] << "(" << this->floatType()
                       << ")(1 << starting_points[(startid + get_global_id(0)) * 2 * " << dimensions
                       << " + 2 * d + 1]);" << std::endl;
          sourceStream << this->indent[1] << "}" << std::endl << std::endl;
          sourceStream << this->indent[0] << "} else {" << std::endl;
          sourceStream << this->indent[1] << "for (int d = 0; d < " << dimensions << "; d++) {"
                       << std::endl;
          sourceStream << this->indent[2] << "grid_index[d] = 1;" << std::endl;
          sourceStream << this->indent[2] << "grid_level_2[d] = 2;" << std::endl;
          sourceStream << this->indent[1] << "}" << std::endl << std::endl;
          sourceStream << this->indent[0] << "}" << std::endl;
        } else {
          sourceStream << this->indent[0] << " ulong one_mask = 1;" << std::endl;
          sourceStream << this->indent[0] << "__private ulong point_dim_zero_flags = 0;"
                       << std::endl;
          sourceStream << this->indent[0] << "__private ulong point_level_offsets = 0;"
                       << std::endl;
          sourceStream << this->indent[0] << "__private ulong point_level_packed = 0;"
                       << std::endl;
          sourceStream << this->indent[0] << "__private ulong point_index_packed = 0;"
                       << std::endl;

          sourceStream << this->indent[0] << "if (get_global_id(0) < " << grid_points << ") {"
                       << std::endl;
          sourceStream << this->indent[1] << "point_dim_zero_flags = dim_zero_flags_v[gridindex];"
                       << std::endl;
          sourceStream << this->indent[1] << "point_level_offsets = level_offsets_v[gridindex];"
                       << std::endl;
          sourceStream << this->indent[1] << "point_level_packed = level_packed_v[gridindex];"
                       << std::endl;
          sourceStream << this->indent[1] << "point_index_packed = index_packed_v[gridindex];"
                       << std::endl;
          sourceStream << this->indent[0] << "}" << std::endl;
        }

        sourceStream << this->indent[0] << "local " << this->floatType() << " data_group["
                     << (local_cache_size * dimensions) << "];" << std::endl
                     << std::endl;

        sourceStream << this->indent[0] << this->floatType() << " result = 0.0" << this->constSuffix()
                     << ";" << std::endl;
        sourceStream << this->indent[0] << "for (int outer_data_index = 0; outer_data_index < "
                     << data_points << ";" << std::endl;
        sourceStream << this->indent[2] << "outer_data_index += " << local_cache_size << ") {"
                     << std::endl;
        sourceStream << this->indent[1] << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
        sourceStream << this->indent[1] << "if (get_local_id(0) < " << local_cache_size << ") {"
                     << std::endl;
        sourceStream << this->indent[2] << "for (int d = 0; d < " << dimensions << "; d++) {"
                     << std::endl;
        sourceStream << this->indent[3] << "int data_index = outer_data_index + get_local_id(0);"
                     << std::endl;
        sourceStream << this->indent[3] << "if (data_index < " << data_points << ") {" << std::endl;
        sourceStream << this->indent[4] << "data_group[get_local_id(0) * " << dimensions
                     << " + d] = data_points[data_index * " << dimensions << " + d];" << std::endl;
        sourceStream << this->indent[3] << "}" << std::endl;
        sourceStream << this->indent[2] << "}" << std::endl;
        sourceStream << this->indent[1] << "}" << std::endl;
        sourceStream << this->indent[1] << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl << std::endl;

        sourceStream << this->indent[1] << "for (int inner_data_index = 0; inner_data_index < "
                     << local_cache_size
                     << "; "
            "inner_data_index += 1) {"
                     << std::endl;
        sourceStream << this->indent[2] << "int data_index = outer_data_index + inner_data_index;"
                     << std::endl;
        sourceStream << this->indent[2] << "if (data_index >= " << data_points << ") {" << std::endl;
        sourceStream << this->indent[3] << "break;" << std::endl;
        sourceStream << this->indent[2] << "}" << std::endl;
        if (use_compression_fixed) {
          sourceStream << this->indent[2] << "ulong fixed_dim_zero_flags = point_dim_zero_flags;" << std::endl;
          sourceStream << this->indent[2] << "ulong fixed_level_offsets = point_level_offsets;" << std::endl;
          sourceStream << this->indent[2] << "ulong fixed_level_packed = point_level_packed;" << std::endl;
          sourceStream << this->indent[2] << "ulong fixed_index_packed = point_index_packed;" << std::endl;
        }
        sourceStream << this->indent[2] << this->floatType() << " eval = 1.0" << this->constSuffix()
                     << ";" << std::endl;
        sourceStream << this->indent[2] << "for (int d = 0; d < " << dimensions << "; d++) {"
                     << std::endl;

        std::string index_func =
            std::string("grid_index[d]");
        std::string level_func =
            std::string("grid_level_2[d]");
        if (use_compression_fixed) {
          sourceStream << this->indent[3]
                       << "ulong is_dim_implicit = fixed_dim_zero_flags & one_mask;" << std::endl;
          sourceStream << this->indent[3] << "fixed_dim_zero_flags >>= 1;" << std::endl;
          sourceStream << this->indent[3] << "ulong decompressed_level = 1;" << std::endl;
          sourceStream << this->indent[3] << "ulong decompressed_index = 1;" << std::endl;
          sourceStream << this->indent[3] << "if (is_dim_implicit != 0) {" << std::endl;
          sourceStream << this->indent[4] << "ulong level_bits = 1 + "
                       << "clz(fixed_level_offsets);"
                       << std::endl;
          sourceStream << this->indent[4] << "fixed_level_offsets <<= level_bits;" << std::endl;
          sourceStream << this->indent[4] << "ulong level_mask = (1 << level_bits) - 1;" << std::endl;
          sourceStream << this->indent[4] << "decompressed_level = (fixed_level_packed & level_mask) + 2;" << std::endl;
          sourceStream << this->indent[4] << "fixed_level_packed >>= level_bits;" << std::endl;
          sourceStream << this->indent[4] << "ulong index_bits = decompressed_level - 1;" << std::endl;
          sourceStream << this->indent[4] << "ulong index_mask = (1 << index_bits) - 1;" << std::endl;
          sourceStream << this->indent[4] << "decompressed_index = ((fixed_index_packed & index_mask) << 1) + 1;" << std::endl;
          sourceStream << this->indent[4] << "fixed_index_packed >>= index_bits;" << std::endl;
          sourceStream << this->indent[3] << "}" << std::endl;
          level_func =
              std::string("decompressed_level");
          index_func =
              std::string("decompressed_index");
          sourceStream << this->indent[3] << this->floatType() << " eval_1d = (" << this->floatType()
                       << ")(1 << " << level_func << ");"
                       << std::endl;
        } else {
          sourceStream << this->indent[3] << this->floatType() << " eval_1d = " << level_func << ";"
                       << std::endl;
        }
        sourceStream << this->indent[3] << "eval_1d *= data_group[inner_data_index * " << dimensions
                     << " + d];" << std::endl;
        sourceStream << this->indent[3] << "eval_1d -= " << index_func << ";" << std::endl;
        sourceStream << this->indent[3] << "eval_1d = fabs(eval_1d);" << std::endl;
        sourceStream << this->indent[3] << "eval_1d = 1 - eval_1d;" << std::endl;
        sourceStream << this->indent[3] << "if (eval_1d < 0) eval_1d = 0;" << std::endl;
        sourceStream << this->indent[3] << "eval *= eval_1d;" << std::endl;
        sourceStream << this->indent[2] << "}" << std::endl;
        sourceStream << this->indent[2] << "result += eval;" << std::endl;
        sourceStream << this->indent[1] << "}" << std::endl;
        sourceStream << this->indent[0] << "}" << std::endl;
        sourceStream << this->indent[0] << "result /= " << data_points << ".0" << this->constSuffix()
                     << ";" << std::endl;
        sourceStream << this->indent[0] << "if (get_global_id(0) < " << grid_points << ") {"
                     << std::endl;
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
