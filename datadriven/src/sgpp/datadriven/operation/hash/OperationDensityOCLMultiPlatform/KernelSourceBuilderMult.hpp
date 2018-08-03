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

/// OpenCL source builder for density matrix vector multiplication
template <typename real_type>
class SourceBuilderMult : public base::KernelSourceBuilderBase<real_type> {
 private:
  /// OpenCL configuration containing the building flags
  json::Node &kernelConfiguration;
  /// Dimensions of grid
  size_t dims;
  /// Used workgroupsize for opencl kernel execution
  size_t localWorkgroupSize;
  /// Using local memory?
  bool useLocalMemory;
  size_t localCacheSize;
  size_t dataBlockSize;
  // size_t transGridBlockSize;
  // uint64_t maxDimUnroll;

  /// Use a cache for the 2^l values? Configuration parameter is USE_LEVEL_CACHE
  bool use_level_cache;
  /// Use a calculation scheme with less operations but more branching?
  bool use_less;
  /// Use ternary operator for branching? Configuration parameter is USE_LESS_OPERATIONS
  bool do_not_use_ternary;
  /// Avoid branching even at the cost of more operations? Configuration parameter is USE_IMPLICIT
  bool use_implicit_zero;
  /// Avoid using fmax? Configuration parameter is USE_FABS
  bool use_fabs_instead_of_fmax;
  /// Use preprocessed grid positions? Configuration parameter is PREPROCESSED_POSITIONS
  bool preprocess_positions;
  uint64_t eval_blocking;
  bool unroll_dim;
  bool use_compression_fixed;
  bool use_compression_streaming;
  bool use_compression_register;
  std::string compression_type;

  /// Generate the opencl code to save the fixed gridpoint of a workitem to the local memory
  std::string save_from_global_to_private(size_t dimensions) {
    std::stringstream output;
    for (size_t block = 0; block < dataBlockSize; block++) {
      if (!preprocess_positions) {
        if (!use_compression_fixed) {
          output << this->indent[0] << "__private int point_indices_block" << block << "["
                 << dimensions << "];" << std::endl;
          output << this->indent[0] << "__private int point_level_block" << block << "["
                 << dimensions << "];" << std::endl;
          for (size_t i = 0; i < dimensions; i++) {
            output << this->indent[0] << "point_indices_block" << block << "[" << i
                   << "] = starting_points[(gridindex * " << dataBlockSize << " + " << block
                   << ") * " << dimensions << " * 2 + 2 * " << i << "];" << std::endl;
            output << this->indent[0] << "point_level_block" << block << "[" << i
                   << "] = starting_points[(gridindex * " << dataBlockSize << " + " << block
                   << ") * " << dimensions << " * 2 + 2 * " << i << " + 1];" << std::endl;
          }
        } else if (use_compression_register) {
          output << this->indent[0] << "__private " << compression_type
                 << " point_dim_zero_flags = dim_zero_flags_v[gridindex];" << std::endl;
          output << this->indent[0] << "__private " << compression_type
                 << " point_level_offsets = level_offsets_v[gridindex];" << std::endl;
          output << this->indent[0] << "__private " << compression_type
                 << " point_level_packed = level_packed_v[gridindex];" << std::endl;
          output << this->indent[0] << "__private " << compression_type
                 << " point_index_packed = index_packed_v[gridindex];" << std::endl;
        }
      } else {
        output << this->indent[0] << "__private " << this->floatType() << " point_positions_block"
               << block << "[" << dimensions << "];" << std::endl;
        output << this->indent[0] << "__private " << this->floatType() << " point_hs_block" << block
               << "[" << dimensions << "];" << std::endl;
        output << this->indent[0] << "__private " << this->floatType() << " point_hinverses_block"
               << block << "[" << dimensions << "];" << std::endl;
        for (size_t i = 0; i < dimensions; i++) {
          // calculate inverses
          output << this->indent[1] << "point_hinverses_block" << block << "[" << i
                 << "] = hs_inverses[gridindex * " << dimensions << " + " << i << "];" << std::endl;
          // calculate hs
          output << this->indent[1] << "point_hs_block" << block << "[" << i
                 << "] = hs[gridindex * " << dimensions << " + " << i << "];" << std::endl;
          // calculate positions
          output << this->indent[1] << "point_positions_block" << block << "[" << i
                 << "] = positions[gridindex * " << dimensions << " + " << i << "];" << std::endl;
        }
      }
    }
    return output.str();
  }

  /// Generates the part of the opencl source code that calculates one entry of the density matrix
  std::string calculate_matrix_entry(size_t block, size_t dimensions) {
    std::stringstream output;
    // Use alias names for levels and indices
    std::string level_func1 =
        std::string("point_level_block") + std::to_string(block) + std::string("[dim]");
    std::string level_func2 = std::string("starting_points[(local_index + j)* ") +
                              std::to_string(dimensions) + std::string("*2+2*dim+1]");
    std::string index_func1 =
        std::string("point_indices_block") + std::to_string(block) + std::string("[dim]");
    std::string index_func2 = std::string("starting_points[(local_index + j)* ") +
                              std::to_string(dimensions) + std::string("*2+2*dim]");
    // In case we use local memory we need to adjust the alias names
    // if (useLocalMemory) {
    if (useLocalMemory && !use_compression_streaming && use_compression_fixed) {
      level_func2 = std::string("level_local[(local_index + j)* ") + std::to_string(dimensions) +
                    std::string("+dim]");
      index_func2 = std::string("indices_local[(local_index + j)* ") + std::to_string(dimensions) +
                    std::string("+dim]");
    } else if (useLocalMemory && !use_compression_streaming && !use_compression_fixed) {
      level_func2 = std::string("level_local[(local_index)* ") + std::to_string(dimensions) +
                    std::string("+dim]");
      index_func2 = std::string("indices_local[(local_index)* ") + std::to_string(dimensions) +
                    std::string("+dim]");
    }
    if (use_compression_fixed) {
      output << this->floatType() << " zellenintegral_blocked[" << eval_blocking << "];"
             << std::endl;
      output << "for (size_t j = 0; j < " << eval_blocking << "; j++) {" << std::endl;
      output << this->indent[1] << "zellenintegral_blocked[j] = 1.0" << this->constSuffix() << ";"
             << std::endl;
      output << "}" << std::endl;
    } else {
      output << " zellenintegral = 1.0" << this->constSuffix() << ";" << std::endl;
    }
    // copy variables for shifting
    if (use_compression_streaming) {
      if (useLocalMemory) {
        output << this->indent[2] << compression_type
               << " current_dim_zero_flags = dim_zero_flags[(local_index + j)];" << std::endl;
        output << this->indent[2] << compression_type
               << " current_level_offsets = level_offsets[(local_index + j)];" << std::endl;
        output << this->indent[2] << compression_type
               << " current_level_packed = level_packed[(local_index + j)];" << std::endl;
        output << this->indent[2] << compression_type
               << " current_index_packed = index_packed[(local_index + j)];" << std::endl;
      } else {
        output << this->indent[2] << compression_type
               << " current_dim_zero_flags = dim_zero_flags_v[(local_index + j)];" << std::endl;
        output << this->indent[2] << compression_type
               << " current_level_offsets = level_offsets_v[(local_index + j)];" << std::endl;
        output << this->indent[2] << compression_type
               << " current_level_packed = level_packed_v[(local_index + j)];" << std::endl;
        output << this->indent[2] << compression_type
               << " current_index_packed = index_packed_v[(local_index + j)];" << std::endl;
      }
    }
    if (use_compression_fixed) {
      if (use_compression_register) {
        output << this->indent[2] << compression_type
               << " fixed_dim_zero_flags = point_dim_zero_flags;" << std::endl;
        output << this->indent[2] << compression_type
               << " fixed_level_offsets = point_level_offsets;" << std::endl;
        output << this->indent[2] << compression_type << " fixed_level_packed = point_level_packed;"
               << std::endl;
        output << this->indent[2] << compression_type << " fixed_index_packed = point_index_packed;"
               << std::endl;
      } else {
        output << this->indent[2] << compression_type
               << " fixed_dim_zero_flags = dim_zero_flags_v[gridindex];" << std::endl;
        output << this->indent[2] << compression_type
               << " fixed_level_offsets = level_offsets_v[gridindex];" << std::endl;
        output << this->indent[2] << compression_type
               << " fixed_level_packed = level_packed_v[gridindex];" << std::endl;
        output << this->indent[2] << compression_type
               << " fixed_index_packed = index_packed_v[gridindex];" << std::endl;
      }
    }
    // In case we want replace the ternary operator we need to declare the counter variable
    if ((use_less && do_not_use_ternary) || (!use_implicit_zero && do_not_use_ternary))
      output << this->indent[2] << "int same_levels = " << dimensions << ";" << std::endl;
    // Loop over all dimensions
    output << this->indent[2] << "for(private int dim = 0;dim< " << dimensions << ";dim++) {"
           << std::endl;

    if (use_compression_fixed) {
      if (!use_compression_streaming) {
        output << this->indent[3] << compression_type
               << " is_dim_implicit = fixed_dim_zero_flags & one_mask;" << std::endl;
      } else {
        output << this->indent[3] << "is_dim_implicit = fixed_dim_zero_flags & one_mask;"
               << std::endl;
      }
      output << this->indent[3] << "fixed_dim_zero_flags >>= 1;" << std::endl;
      output << this->indent[3] << compression_type << " decompressed_level = 1;" << std::endl;
      output << this->indent[3] << compression_type << " decompressed_index = 1;" << std::endl;
      output << this->indent[3] << "if (is_dim_implicit != 0) {" << std::endl;
      output << this->indent[4] << compression_type << " level_bits = 1 + "
             << "clz(fixed_level_offsets);" << std::endl;
      output << this->indent[4] << "fixed_level_offsets <<= level_bits;" << std::endl;
      output << this->indent[4] << compression_type << " level_mask = (1 << level_bits) - 1;"
             << std::endl;
      output << this->indent[4] << "decompressed_level = (fixed_level_packed & level_mask) + 2;"
             << std::endl;
      output << this->indent[4] << "fixed_level_packed >>= level_bits;" << std::endl;
      output << this->indent[4] << compression_type << " index_bits = decompressed_level - 1;"
             << std::endl;
      output << this->indent[4] << compression_type << " index_mask = (1 << index_bits) - 1;"
             << std::endl;
      output << this->indent[4]
             << "decompressed_index = ((fixed_index_packed & index_mask) << 1) + 1;" << std::endl;
      output << this->indent[4] << "fixed_index_packed >>= index_bits;" << std::endl;
      output << this->indent[3] << "}" << std::endl;
      output << this->indent[3] << "float l_2 = (float)(1 << decompressed_level);" << std::endl;
      output << this->indent[3] << "float i = (float)(decompressed_index);" << std::endl;
      output << this->indent[3] << "for (size_t j = 0; j < " << eval_blocking << "; j++) {"
             << std::endl;
      level_func1 = std::string("l_2");
      index_func1 = std::string("i");
    }
    // if we use compression - now is the time to decompress
    if (use_compression_streaming) {
      output << this->indent[3] << compression_type
             << " is_dim_implicit = current_dim_zero_flags & one_mask;" << std::endl;
      output << this->indent[3] << "current_dim_zero_flags >>= 1;" << std::endl;
      output << this->indent[3] << compression_type << " decompressed_level2 = 1;" << std::endl;
      output << this->indent[3] << compression_type << " decompressed_index2 = 1;" << std::endl;
      output << this->indent[3] << "if (is_dim_implicit != 0) {" << std::endl;
      output << this->indent[4] << compression_type << " level_bits = 1 + "
             << "clz(current_level_offsets);" << std::endl;
      output << this->indent[4] << "current_level_offsets <<= level_bits;" << std::endl;
      output << this->indent[4] << compression_type << " level_mask = (1 << level_bits) - 1;"
             << std::endl;
      output << this->indent[4] << "decompressed_level2 = (current_level_packed & level_mask) + 2;"
             << std::endl;
      output << this->indent[4] << "current_level_packed >>= level_bits;" << std::endl;
      output << this->indent[4] << compression_type << " index_bits = decompressed_level2 - 1;"
             << std::endl;
      output << this->indent[4] << compression_type << " index_mask = (1 << index_bits) - 1;"
             << std::endl;
      output << this->indent[4]
             << "decompressed_index2 = ((current_index_packed & index_mask) << 1) + 1;"
             << std::endl;
      output << this->indent[4] << "current_index_packed >>= index_bits;" << std::endl;
      output << this->indent[3] << "}" << std::endl;
      level_func2 = std::string("decompressed_level2");
      index_func2 = std::string("decompressed_index2");
    }
    // In case we do not want to use that the entry is implicitly zero if we use the wrong order
    // we need to find the smallest level
    if (!use_implicit_zero) {
      // Initialise with one gridpoint
      output << this->indent[3] << "int index = " << index_func1 << ";" << std::endl;
      output << this->indent[3] << "int level = " << level_func1 << ";" << std::endl;
      output << this->indent[3] << "int index2 = " << index_func2 << ";" << std::endl;
      output << this->indent[3] << "int level2 = " << level_func2 << ";" << std::endl;
      // Check whether we need to use the other gridpoint as gridpoint 1
      output << this->indent[3] << "if (level > level2) {" << std::endl;
      output << this->indent[4] << "index = " << index_func2 << ";" << std::endl;
      output << this->indent[4] << "level = " << level_func2 << ";" << std::endl;
      output << this->indent[4] << "index2 = " << index_func1 << ";" << std::endl;
      output << this->indent[4] << "level2 = " << level_func1 << ";" << std::endl;
      output << this->indent[3] << "}" << std::endl;
      // Replace alias names
      level_func1 = std::string("level");
      level_func2 = std::string("level2");
      index_func1 = std::string("index");
      index_func2 = std::string("index2");
    }
    // Loop to evaluate the base function on the left, right or mid points of the other basefunction
    for (int i = 0; i < 1 + static_cast<int>(use_implicit_zero); i++) {
      if (use_level_cache) {
        // Reuse h values from host
        output << this->indent[3] << "h = hs[" << level_func2 << "];" << std::endl;
      } else {
        // Calculate h
        if (level_func2 == std::string("l_2"))
          output << this->indent[3] << "h = 1.0 / (" << level_func2 << ");" << std::endl;
        else
          output << this->indent[3] << "h = 1.0 / (1 << " << level_func2 << ");" << std::endl;
      }
      // Calculate u
      if (level_func1 == std::string("l_2"))
        output << this->indent[3] << "u = (" << level_func1 << ");" << std::endl;
      else
        output << this->indent[3] << "u = (1 << " << level_func1 << ");" << std::endl;
      // Check whether we will just need to calculate umid, or umid uright and uleft
      if (use_less) {
        // Calculate just umid
        output << this->indent[3] << "umid = u * h * (" << index_func2 << ") - " << index_func1
               << ";" << std::endl;
        output << this->indent[3] << "umid = fabs(umid);" << std::endl;
        output << this->indent[3] << "umid = 1.0-umid;" << std::endl;
        if (!use_fabs_instead_of_fmax)
          output << this->indent[3] << "umid = fmax(umid,0.0);" << std::endl;
        else
          output << this->indent[3] << "umid = (umid + fabs(umid));" << std::endl;
        // Add integral to result sum
        if (i == 0)
          output << this->indent[3] << "sum = h*(umid);" << std::endl;
        else
          output << this->indent[3] << "sum += h*(umid);" << std::endl;
      } else {
        // Calculate umid, uright and uleft
        output << this->indent[3] << "umid = u * h * (" << index_func2 << ") - " << index_func1
               << ";" << std::endl;
        output << this->indent[3] << "umid = fabs(umid);" << std::endl;
        output << this->indent[3] << "umid = 1.0-umid;" << std::endl;
        if (!use_fabs_instead_of_fmax)
          output << this->indent[3] << "umid = fmax(umid,0.0);" << std::endl;
        else
          output << this->indent[3] << "umid = (umid + fabs(umid));" << std::endl;
        output << this->indent[3] << "uright = u*h*(" << index_func2 << " + 1) - " << index_func1
               << ";" << std::endl;
        output << this->indent[3] << "uright = fabs(uright);" << std::endl;
        output << this->indent[3] << "uright = 1.0-uright;" << std::endl;
        if (!use_fabs_instead_of_fmax)
          output << this->indent[3] << "uright = fmax(uright,0.0);" << std::endl;
        else
          output << this->indent[3] << "uright = (uright + fabs(uright));" << std::endl;
        output << this->indent[3] << "uleft = u*h*(" << index_func2 << " - 1) - " << index_func1
               << ";" << std::endl;
        output << this->indent[3] << "uleft = fabs(uleft);" << std::endl;
        output << this->indent[3] << "uleft = 1.0-uleft;" << std::endl;
        if (!use_fabs_instead_of_fmax)
          output << this->indent[3] << "uleft = fmax(uleft,0.0);" << std::endl;
        else
          output << this->indent[3] << "uleft = (uleft + fabs(uleft));" << std::endl;
        // Add integral to result sum
        if (i == 0)
          output << this->indent[3] << "sum = h/3.0*(umid + uleft + uright);" << std::endl;
        else
          output << this->indent[3] << "sum += h/3.0*(umid + uleft + uright);" << std::endl;
      }
      // Swap aliases
      std::string level_func_tmp = level_func2;
      level_func2 = level_func1;
      level_func1 = level_func_tmp;
      std::string index_func_tmp = index_func2;
      index_func2 = index_func1;
      index_func1 = index_func_tmp;
    }
    // Check whether we need to do something about base functions with the same level
    if (use_less) {
      if (!do_not_use_ternary) {
        if (use_implicit_zero) {
          // Use ternary operator to multiply with 1/3
          if (use_compression_fixed)
            output << this->indent[3] << "sum *= " << level_func2 << " == "
                   << "decompressed_level ? 1.0/3.0 : 1.0;" << std::endl;
          else
            output << this->indent[3] << "sum *= " << level_func2 << " == " << level_func1
                   << "? 1.0/3.0 : 1.0;" << std::endl;
        } else {
          // Use ternary operator to multiply with 2/3
          output << this->indent[3] << "sum *= " << level_func2 << " == " << level_func1
                 << " ? 2.0/3.0 : 1.0;" << std::endl;
        }
      } else {
        // decrement counter of same levels by one if the two levels do not match
        output << this->indent[3] << "same_levels -= min((int)(abs(level_local[(local_index + j)*"
               << dimensions << "+dim] - point_level_block" << block << "[dim])),(int)(1));"
               << std::endl;
      }
    } else if (!use_implicit_zero) {
      if (!do_not_use_ternary) {
        output << this->indent[3] << "sum *= " << level_func2 << " == " << level_func1
               << " ? 2.0 : 1.0;" << std::endl;
      } else {
        // decrement counter of same levels by one if the two levels do not match
        output << this->indent[3] << "same_levels -= min((int)(abs(level_local[(local_index + j)*"
               << dimensions << "+dim] - point_level_block" << block << "[dim])),(int)(1));"
               << std::endl;
      }
    }
    // Update cell integral
    if (!use_compression_fixed) {
      output << this->indent[3] << "zellenintegral*=sum;" << std::endl;
    } else {
      output << this->indent[3] << "zellenintegral_blocked[j]*=sum;" << std::endl;
      output << this->indent[2] << "}" << std::endl; // CHANGED: correct?
    }
    output << this->indent[2] << "}" << std::endl;
    // Update cell integral with missing factors
    if (do_not_use_ternary) {
      if (use_less) {
        output << this->indent[2] << "zellenintegral *= divisors[same_levels];" << std::endl;
        if (!use_implicit_zero) {
          output << this->indent[2] << "zellenintegral *= (1 << same_levels);" << std::endl;
        }
      } else if (!use_implicit_zero) {
        output << this->indent[2] << "zellenintegral *= (1 << same_levels);" << std::endl;
      }
    }
    // Mutliply with corresponding alpha value
    if (use_compression_streaming) {
      if (useLocalMemory)
        output << this->indent[2] << "if (group * " << localCacheSize << " + i < non_padding_size)"
               << std::endl;
      else
        output << this->indent[2] << "if (i < non_padding_size)" << std::endl;
    }
    if (useLocalMemory) {
      if (use_compression_fixed) {
        output << "for (size_t j = 0; j < " << eval_blocking << "; j++) {" << std::endl;
        output << this->indent[2] << "gesamtint_block" << block
               << " += zellenintegral_blocked[j]*alpha_local[(local_index + j)];" << std::endl;
        output << "}" << std::endl;
      } else {
        output << this->indent[2] << "gesamtint_block" << block
               << " += zellenintegral*alpha_local[(local_index)];" << std::endl;
      }
    } else {
      output << this->indent[2] << "gesamtint_block" << block
             << " += zellenintegral*alpha[(local_index)];" << std::endl;
    }
    return output.str();
  }

 public:
  explicit SourceBuilderMult(json::Node &kernelConfiguration)
      : kernelConfiguration(kernelConfiguration),
        dataBlockSize(1),
        use_level_cache(false),
        use_less(true),
        do_not_use_ternary(false),
        use_implicit_zero(true),
        use_fabs_instead_of_fmax(false),
        preprocess_positions(false),
        unroll_dim(false) {
    // if (kernelConfiguration.contains("LOCAL_SIZE"))
    localWorkgroupSize = kernelConfiguration["LOCAL_SIZE"].getUInt();
    // if (kernelConfiguration.contains("KERNEL_USE_LOCAL_MEMORY"))
    useLocalMemory = kernelConfiguration["KERNEL_USE_LOCAL_MEMORY"].getBool();
    // if (kernelConfiguration.contains("KERNEL_LOCAL_MEMORY_CACHE_SIZE"))
    localCacheSize = kernelConfiguration["KERNEL_LOCAL_CACHE_SIZE"].getUInt();
    // if (kernelConfiguration.contains("KERNEL_DATA_BLOCKING_SIZE"))
    dataBlockSize = kernelConfiguration["KERNEL_DATA_BLOCKING_SIZE"].getUInt();
    // if (kernelConfiguration.contains("USE_LEVEL_CACHE"))
    use_level_cache = kernelConfiguration["USE_LEVEL_CACHE"].getBool();
    // if (kernelConfiguration.contains("USE_LESS_OPERATIONS"))
    use_less = kernelConfiguration["USE_LESS_OPERATIONS"].getBool();
    // if (kernelConfiguration.contains("DO_NOT_USE_TERNARY"))
    do_not_use_ternary = kernelConfiguration["DO_NOT_USE_TERNARY"].getBool();
    // if (kernelConfiguration.contains("USE_IMPLICIT"))
    use_implicit_zero = kernelConfiguration["USE_IMPLICIT"].getBool();
    // if (kernelConfiguration.contains("USE_FABS"))
    use_fabs_instead_of_fmax = kernelConfiguration["USE_FABS"].getBool();
    // if (kernelConfiguration.contains("PREPROCESS_POSITIONS")) {
    preprocess_positions = kernelConfiguration["PREPROCESS_POSITIONS"].getBool();
    eval_blocking = kernelConfiguration["KERNEL_EVAL_BLOCKING"].getUInt();

    if (kernelConfiguration.contains("USE_COMPRESSION_STREAMING"))
      use_compression_streaming = kernelConfiguration["USE_COMPRESSION_STREAMING"].getBool();
    else
      use_compression_streaming = false;
    if (kernelConfiguration.contains("USE_COMPRESSION_FIXED"))
      use_compression_fixed = kernelConfiguration["USE_COMPRESSION_FIXED"].getBool();
    else
      use_compression_fixed = false;

    if (kernelConfiguration.contains("COMPRESSION_TYPE")) {
      if (kernelConfiguration["COMPRESSION_TYPE"].get().compare("uint64_t") == 0) {
        compression_type = "ulong";
      } else if (kernelConfiguration["COMPRESSION_TYPE"].get().compare("unsigned int") == 0) {
        compression_type = "unsigned int";
      } else {
        throw base::operation_exception(
            "OCL error: Illegal value for parameter \"COMPRESSION_TYPE\"\n");
      }
    } else {
      compression_type = "ulong";
    }
    if (kernelConfiguration.contains("USE_COMPRESSION_REGISTERS")) {
      use_compression_register = kernelConfiguration["USE_COMPRESSION_REGISTERS"].getBool();
    } else {
      use_compression_register = true;
    }

    // These two options are not compatible
    if (preprocess_positions) use_level_cache = false;
    // }
    // if (kernelConfiguration.contains("UNROLL_DIM")) {
    unroll_dim = kernelConfiguration["UNROLL_DIM"].getBool();
    // }
  }

  /// Generates the opencl source code for the density matrix-vector multiplication
  std::string generateSource(size_t dimensions, size_t problemsize) {
    if (kernelConfiguration["REUSE_SOURCE"].getBool()) {
      return this->reuseSource("DensityMultiplication.cl");
    }

    std::stringstream sourceStream;

    if (this->floatType().compare("double") == 0) {
      sourceStream << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" << std::endl << std::endl;
    }

    sourceStream << "__kernel" << std::endl;
    sourceStream << "__attribute__((reqd_work_group_size(" << localWorkgroupSize << ", 1, 1)))"
                 << std::endl;
    if (!preprocess_positions) {
      if (use_compression_streaming || use_compression_fixed) {
        sourceStream << "void multdensity(__global const int *starting_points, "
                     << "__global const " << compression_type << " *dim_zero_flags_v, "
                     << "__global const " << compression_type << " *level_offsets_v, "
                     << "__global const " << compression_type << " *level_packed_v, "
                     << "__global const " << compression_type << " *index_packed_v, "
                     << "const unsigned int non_padding_size, ";
      } else {
        sourceStream << "void multdensity(__global const int *starting_points,";
      }
    } else {
      sourceStream << "void multdensity(__global const int *hs_inverses, __global const "
                   << this->floatType() << " *hs," << std::endl
                   << " __global const " << this->floatType() << " *positions,";
    }
    sourceStream << "__global const " << this->floatType() << " *alpha, __global "
                 << this->floatType() << " *result,const " << this->floatType()
                 << " lambda, const int startid";
    if (use_level_cache) sourceStream << ", __global " << this->floatType() << " *hs";
    if (do_not_use_ternary) sourceStream << ", __global " << this->floatType() << " *divisors";
    sourceStream << ")" << std::endl;
    sourceStream << "{" << std::endl;
    sourceStream << this->indent[0] << "int gridindex = startid + get_global_id(0);" << std::endl;
    if (use_compression_streaming || use_compression_fixed) {
      // sourceStream << this->indent[2] << "if(gridindex >= non_padding_size) return;" <<
      // std::endl;
      sourceStream << this->indent[0] << " " << compression_type << " one_mask = 1;" << std::endl;
    }
    sourceStream << this->indent[0] << "__private int local_id = get_local_id(0);" << std::endl;
    sourceStream << save_from_global_to_private(dimensions);
    sourceStream << this->indent[0] << "__private int teiler = 0;" << std::endl;
    sourceStream << this->indent[0] << "__private " << this->floatType() << " h = 1.0 / 3.0;"
                 << std::endl;
    sourceStream << this->indent[0] << "__private " << this->floatType() << " umid = 0.0;"
                 << std::endl;
    if (!use_less) {
      sourceStream << this->indent[0] << "__private " << this->floatType() << " uright = 0.0;"
                   << std::endl;
      sourceStream << this->indent[0] << "__private " << this->floatType() << " uleft = 0.0;"
                   << std::endl;
    }
    sourceStream << this->indent[0] << "__private " << this->floatType() << " sum = 0.0;"
                 << std::endl;
    sourceStream << this->indent[0] << "__private int u= 0;" << std::endl;
    for (size_t block = 0; block < dataBlockSize; block++)
      sourceStream << this->indent[0] << this->floatType() << " gesamtint_block" << block
                   << " = 0.0;" << std::endl;
    // Store points in local memory
    if (useLocalMemory && !preprocess_positions) {
      if (!use_compression_streaming) {
        sourceStream << this->indent[0] << "__local "
                     << "int indices_local[" << localCacheSize * dimensions << "];" << std::endl
                     << this->indent[0] << "__local "
                     << "int level_local[" << localCacheSize * dimensions << "];" << std::endl
                     << this->indent[0] << "__local " << this->floatType() << " alpha_local["
                     << localCacheSize << "];" << std::endl
                     << this->indent[0] << "for (int group = 0; group < "
                     << problemsize / localCacheSize << "; group++) {" << std::endl
                     << this->indent[1] << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
        sourceStream << this->indent[1] << "if (get_local_id(0) < " << localCacheSize << ") {"
                     << std::endl;
        sourceStream << this->indent[1] << "for (int j = 0; j <     " << dimensions << " ; j++) {"
                     << std::endl
                     << this->indent[2] << "indices_local[local_id * " << dimensions
                     << " + j] = starting_points[group * " << localCacheSize * dimensions * 2
                     << "  + local_id * " << dimensions * 2 << " + 2 * j];" << std::endl
                     << this->indent[2] << "level_local[local_id * " << dimensions
                     << " + j] = starting_points[group * " << localCacheSize * dimensions * 2
                     << "  + local_id * " << dimensions * 2 << " + 2 * j + 1];" << std::endl
                     << this->indent[1] << "}" << std::endl
                     << this->indent[1] << "alpha_local[local_id] = alpha[group * "
                     << localCacheSize << "  + local_id ];" << std::endl;
        sourceStream << this->indent[1] << "}" << std::endl
                     << this->indent[1] << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
        sourceStream << this->indent[1] << "for (int local_index = 0; local_index < "
                     << localCacheSize << "; local_index += " << eval_blocking << ") {"
                     << std::endl;
        if (!use_compression_fixed)
          sourceStream << this->indent[2] << "__private " << this->floatType();
      } else {  // use compression

        sourceStream << this->indent[0] << "__local " << compression_type << " dim_zero_flags["
                     << localCacheSize << "];" << std::endl
                     << this->indent[0] << "__local " << compression_type << " level_offsets["
                     << localCacheSize << "];" << std::endl
                     << this->indent[0] << "__local " << compression_type << " level_packed["
                     << localCacheSize << "];" << std::endl
                     << this->indent[0] << "__local " << compression_type << " index_packed["
                     << localCacheSize << "];" << std::endl
                     << this->indent[0] << "__local " << this->floatType() << " alpha_local["
                     << localCacheSize << "];" << std::endl
                     << this->indent[0] << "for (int group = 0; group < "
                     << problemsize / localCacheSize << "; group++) {" << std::endl
                     << this->indent[1] << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
        sourceStream << this->indent[1] << "if (get_local_id(0) < " << localCacheSize << ") {"
                     << std::endl;
        sourceStream << this->indent[1] << "dim_zero_flags[local_id]"
                     << " = dim_zero_flags_v[group * " << localCacheSize << "  + local_id];"
                     << std::endl;
        sourceStream << this->indent[1] << "level_offsets[local_id]"
                     << " = level_offsets_v[group * " << localCacheSize << "  + local_id];"
                     << std::endl;
        sourceStream << this->indent[1] << "level_packed[local_id]"
                     << " = level_packed_v[group * " << localCacheSize << "  + local_id];"
                     << std::endl;
        sourceStream << this->indent[1] << "index_packed[local_id]"
                     << " = index_packed_v[group * " << localCacheSize << "  + local_id];"
                     << std::endl;

        sourceStream << this->indent[1] << "alpha_local[local_id] = alpha[group * "
                     << localCacheSize << "  + local_id ];" << std::endl;
        sourceStream << this->indent[1] << "}" << std::endl
                     << this->indent[1] << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl
                     << this->indent[1] << "for (int local_index = 0; local_index < "
                     << localCacheSize << "; local_index += " << eval_blocking << ") {"
                     << std::endl;
        if (!use_compression_fixed)
          sourceStream << this->indent[2] << "__private " << this->floatType();
      }
    } else if (preprocess_positions && !unroll_dim) {
      // declare local arrays for grid point positions, and hs and hs inverses
      sourceStream << this->indent[0] << "__local " << this->floatType() << " positions_local["
                   << localCacheSize * dimensions << "];" << std::endl
                   << this->indent[0] << "__local " << this->floatType() << " hs_local["
                   << localCacheSize * dimensions << "];" << std::endl
                   << this->indent[0] << "__local int hinverses_local["
                   << localCacheSize * dimensions << "];" << std::endl
                   << this->indent[0] << "__local " << this->floatType() << " alpha_local["
                   << localCacheSize << "];" << std::endl;
      // start loop
      sourceStream << this->indent[0] << "for (int group = 0; group < "
                   << problemsize / localCacheSize << "; group++) {" << std::endl
                   << this->indent[1] << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
      sourceStream << this->indent[1] << "if (get_local_id(0) < " << localCacheSize << ") {"
                   << std::endl;
      sourceStream << this->indent[1] << "for (int j = 0; j <     " << dimensions << " ; j++) {"
                   << std::endl;
      // get hs inverse
      sourceStream << this->indent[2] << "hinverses_local[local_id * " << dimensions
                   << " + j] = hs_inverses[group * " << localCacheSize * dimensions
                   << " + local_id*" << dimensions << " + j];" << std::endl;
      // get hs
      sourceStream << this->indent[2] << "hs_local[local_id * " << dimensions
                   << " + j] = hs[group * " << localCacheSize * dimensions << " + local_id*"
                   << dimensions << " + j];" << std::endl;
      // get positions
      sourceStream << this->indent[2] << "positions_local[local_id * " << dimensions
                   << " + j] = positions[group * " << localCacheSize * dimensions << " + local_id*"
                   << dimensions << " + j];" << std::endl;
      sourceStream << "}" << this->indent[1] << "alpha_local[local_id] = alpha[group * "
                   << localCacheSize << "  + local_id ];" << std::endl;
      sourceStream << this->indent[1] << "}" << std::endl
                   << this->indent[1] << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl
                   << this->indent[1] << "for (int i = 0 ; i < " << localCacheSize << "; i++) {"
                   << std::endl
                   << this->indent[2] << "__private " << this->floatType();
    } else if (preprocess_positions && unroll_dim) {
      for (size_t dim = 0; dim < dimensions; ++dim) {
        sourceStream << this->indent[0] << "__local " << this->floatType() << " positions_local_dim"
                     << dim << "[" << localCacheSize << "];" << std::endl
                     << this->indent[0] << "__local " << this->floatType() << " hs_local"
                     << "_dim" << dim << "[" << localCacheSize << "];" << std::endl
                     << this->indent[0] << "__local int hinverses_local"
                     << "_dim" << dim << "[" << localCacheSize << "];" << std::endl;
      }
      sourceStream << this->indent[0] << "__local " << this->floatType() << " alpha_local["
                   << localCacheSize << "];" << std::endl;
      // start loop
      sourceStream << this->indent[0] << "for (int group = 0; group < "
                   << problemsize / localCacheSize << "; group++) {" << std::endl
                   << this->indent[1] << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
      sourceStream << this->indent[1] << "if (get_local_id(0) < " << localCacheSize << ") {"
                   << std::endl;
      for (size_t dim = 0; dim < dimensions; ++dim) {
        // get hs inverse
        sourceStream << this->indent[2] << "hinverses_local_dim" << dim
                     << "[local_id] = hs_inverses[group * " << localCacheSize * dimensions
                     << " + local_id*" << dimensions << " + " << dim << "];" << std::endl;
        // get hs
        sourceStream << this->indent[2] << "hs_local_dim" << dim << "[local_id] = hs[group * "
                     << localCacheSize * dimensions << " + local_id*" << dimensions << " + " << dim
                     << "];" << std::endl;
        // get positions
        sourceStream << this->indent[2] << "positions_local_dim" << dim
                     << "[local_id] = positions[group * " << localCacheSize * dimensions
                     << " + local_id*" << dimensions << " + " << dim << "];" << std::endl;
      }
      sourceStream << this->indent[1] << "alpha_local[local_id] = alpha[group * " << localCacheSize
                   << "  + local_id ];" << std::endl;
      sourceStream << this->indent[1] << "}" << std::endl
                   << this->indent[1] << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl
                   << this->indent[1] << "for (int i = 0 ; i < " << localCacheSize << "; i++) {"
                   << std::endl
                   << this->indent[2] << "__private " << this->floatType();
    } else {
      sourceStream << this->indent[1] << "for (int i = 0 ; i < " << problemsize << "; i++) {"
                   << std::endl
                   << this->indent[2] << "__private " << this->floatType();
    }

    // Generate body for each element in the block
    for (size_t block = 0; block < dataBlockSize; block++) {
      if (preprocess_positions) {
        sourceStream << " zellenintegral = 1.0;" << std::endl;
        if (!unroll_dim) {
          // Loop over all dimensions
          sourceStream << this->indent[2] << "for(private int dim = 0;dim< " << dimensions
                       << ";dim++) {" << std::endl;
          // Calculate distance between grid points in this dimensions
          sourceStream << this->indent[2] << "__private " << this->floatType()
                       << " distance = fabs(point_positions_block" << block << "[dim] - "
                       << "positions_local[i * " << dimensions << " + dim]);" << std::endl;
          // Calculate first integral (level_point < level_local)
          sourceStream << this->indent[2] << "sum = 1.0 - distance * point_hinverses_block" << block
                       << "[dim]; " << std::endl;
          sourceStream << this->indent[2] << "sum *= hs_local[i *" << dimensions << " + dim]; "
                       << std::endl;
          sourceStream << this->indent[2] << "sum = max(sum, (" << this->floatType() << ") 0.0); "
                       << std::endl;
          // Calculate second integral (level_point > level_local)
          sourceStream << this->indent[2] << "sum += max((" << this->floatType()
                       << ")(point_hs_block" << block << "[dim] * (1.0 - hinverses_local[i * "
                       << dimensions << " + dim] * distance)), (" << this->floatType() << ")0.0);"
                       << std::endl;
          // Update cell integral
          if (this->floatType().compare("double") == 0) {
            sourceStream << this->indent[3] << "zellenintegral*=sum*select((" << this->floatType()
                         << " )1.0,"
                         << " (" << this->floatType() << " )h, (long)(point_hinverses_block"
                         << block << "[dim] == hinverses_local[i * " << dimensions << " + dim]));"
                         << std::endl;
          } else {
            sourceStream << this->indent[3] << "zellenintegral*=sum*select((" << this->floatType()
                         << " )1.0,"
                         << " (" << this->floatType() << " )h, (uint)(point_hinverses_block"
                         << block << "[dim] == hinverses_local[i * " << dimensions << " + dim]));"
                         << std::endl;
          }
          sourceStream << this->indent[2] << "}" << std::endl;
        } else {
          for (size_t dim = 0; dim < dimensions; ++dim) {
            // Calculate distance between grid points in this dimensions
            sourceStream << this->indent[2] << "__private " << this->floatType() << " distance_dim"
                         << dim << " = fabs(point_positions_block" << block << "[" << dim << "] - "
                         << "positions_local_dim" << dim << "[i]);" << std::endl;
            // Calculate first integral (level_point < level_local)
            sourceStream << this->indent[2] << "sum = 1.0 - distance_dim" << dim
                         << " * point_hinverses_block" << block << "[" << dim << "]; " << std::endl;
            sourceStream << this->indent[2] << "sum *= hs_local_dim" << dim << "[i]; " << std::endl;
            sourceStream << this->indent[2] << "sum = max(sum, (" << this->floatType() << ")0.0); "
                         << std::endl;
            // Calculate second integral (level_point > level_local)
            sourceStream << this->indent[2] << "sum += max((" << this->floatType()
                         << ")(point_hs_block" << block << "[" << dim
                         << "] * (1.0 - hinverses_local_dim" << dim << "[i] * distance_dim" << dim
                         << ")), (" << this->floatType() << ")0.0);" << std::endl;
            // Update cell integral
            if (this->floatType().compare("double") == 0) {
              sourceStream << this->indent[3] << "zellenintegral*=sum*select((" << this->floatType()
                           << " )1.0,"
                           << " (" << this->floatType() << " )h, (long)(point_hinverses_block"
                           << "[" << dim << "] == hinverses_local_dim" << dim << "[i]));"
                           << std::endl;
            } else {
              sourceStream << this->indent[3] << "zellenintegral*=sum*select((" << this->floatType()
                           << " )1.0,"
                           << " (" << this->floatType() << " )h, (uint)(point_hinverses_block"
                           << "[" << dim << "] == hinverses_local_dim" << dim << "[i]));"
                           << std::endl;
            }
          }
        }
        sourceStream << this->indent[2] << "gesamtint_block" << block
                     << " += zellenintegral*alpha_local[i];" << std::endl;
      } else {
        sourceStream << calculate_matrix_entry(block, dimensions) << std::endl;
      }
    }
    // Close group loop
    if (useLocalMemory) sourceStream << this->indent[1] << "}" << std::endl;
    sourceStream << this->indent[0] << "}" << std::endl;
    for (size_t block = 0; block < dataBlockSize; ++block) {
      if (!use_fabs_instead_of_fmax || preprocess_positions)
        sourceStream << this->indent[0] << "result[get_global_id(0) * " << dataBlockSize << " + "
                     << block << "] = gesamtint_block" << block << ";" << std::endl;
      else
        sourceStream << this->indent[0] << "result[get_global_id(0) * " << dataBlockSize << " + "
                     << block << "] = gesamtint_block" << block << " / " << (1 << dimensions) << ";"
                     << std::endl;
      sourceStream << this->indent[0] << "result[get_global_id(0) * " << dataBlockSize << " + "
                   << block << "] += alpha[gridindex * " << dataBlockSize << " + " << block << "]*"
                   << "lambda;" << std::endl;
    }

    sourceStream << "}" << std::endl;

    if (kernelConfiguration["WRITE_SOURCE"].getBool() &&
        !kernelConfiguration["REUSE_SOURCE"].getBool()) {
      this->writeSource("DensityMultiplication.cl", sourceStream.str());
    }
    return sourceStream.str();
  }
};  // namespace DensityOCLMultiPlatform

}  // namespace DensityOCLMultiPlatform
}  // namespace datadriven
}  // namespace sgpp
