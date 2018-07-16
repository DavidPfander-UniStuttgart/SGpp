// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org
#pragma once

#include <vector>
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>

class compressed_grid {
 public:
  compressed_grid(std::vector<int> &gridpoints, size_t dimensions) :
      gridsize(gridpoints.size()/(2 * dimensions)), dims(dimensions),
      dim_zero_flags_v(gridpoints.size()/(2 * dimensions), 0),
      level_offsets_v(gridpoints.size()/(2 * dimensions), 0),
      level_packed_v(gridpoints.size()/(2 * dimensions), 0),
      index_packed_v(gridpoints.size()/(2 * dimensions), 0)
  {

    for (size_t i = 0; i < gridsize; i++) {
      pack_gridpoint(gridpoints, i);
    }
  }
 public:
  std::vector<uint64_t> dim_zero_flags_v;
  std::vector<uint64_t> level_offsets_v;
  std::vector<uint64_t> level_packed_v;
  std::vector<uint64_t> index_packed_v;

  ///
  std::string extraction_ocl_code(std::string dim_zero_flags_name, std::string
                                  level_offset_name, std::string level_packed_name, std::string
                                  index_packed_name) {
    std::stringstream output;
    output << "uint64_t is_dim_implicit = dim_zero_flags & one_mask;" << std::endl;
    output << "dim_zero_flags >>= 1;" << std::endl;
    output << "if (is_dim_implicit == 0) {" << std::endl;
    output << "  level_d = 1;" << std::endl;
    output << "  index_d = 1;" << std::endl;
    output << "  return;" << std::endl;
    output << "}" << std::endl;
    output << "uint64_t level_bits = __builtin_ffs(level_offsets);" << std::endl;
    // uint64_t level_bits = __builtin_ctz(level_offsets) + 1;
    output << "level_offsets >>= level_bits;" << std::endl;
    output << "uint64_t level_mask = (1 << level_bits) - 1;" << std::endl;
    output << "level_d = (level_packed & level_mask) + 2;" << std::endl;
    output << "level_packed >>= level_bits;" << std::endl;
    output << "uint64_t index_bits = level_d - 1;" << std::endl;
    output << "uint64_t index_mask = (1 << index_bits) - 1;" << std::endl;
    output << "index_d = ((index_packed & index_mask) << 1) + 1;" << std::endl;
    output << "index_packed >>= index_bits;" << std::endl;
    return output.str();
  }

  bool check_grid_compression(std::vector<int> &gridpoints) {
    // gridsize = dim_zero_flags_v.size();
    // if (gridsize != gridpoints.size() / (2 * dims)) {
    //   std::cerr << "should be " << gridsize << " is " <<gridpoints.size() / (2 * dims) << std::endl;
    //   return false;
    // }
    for (size_t i = 0; i < gridsize; i++) {
      uint64_t dim_zero_flags_copy = dim_zero_flags_v[i];
      uint64_t level_offsets_copy = level_offsets_v[i];
      uint64_t level_packed_copy = level_packed_v[i];
      uint64_t index_packed_copy = index_packed_v[i];
      for (size_t d = 0; d < dims; d++) {
        uint64_t extracted_level = 0;
        uint64_t extracted_index = 0;
        get_next(extracted_level, extracted_index, dim_zero_flags_copy, level_offsets_copy,
                 level_packed_copy, index_packed_copy);
        const int grid_index = gridpoints[i * 2 * dims + 2 * d];
        const int grid_level = gridpoints[i * 2 * dims + 2 * d + 1];
        if (grid_index != extracted_index || grid_level != extracted_level) {
          std::cerr << "should be " << grid_index << " " << grid_level << std::endl;
          std::cerr << "is " << extracted_index << " " << extracted_level << std::endl;
          return false;
        }
      }
    }
    return true;
  }
 private:
  size_t gridsize;
  size_t dims;
  void pack_gridpoint(std::vector<int> &gridpoints, size_t i) {
    uint64_t level[dims];
    uint64_t index[dims];
    // stage values
    for (int64_t d = 0; d < dims; d++) {
      index[d] = gridpoints[i * 2 * dims + 2 * d];
      level[d] = gridpoints[i * 2 * dims + 2 * d + 1];
    }
    // pack values into the compressed vectors
    for (int64_t d = dims - 1; d >= 0; d--) {
      pack_level_index_dim(level, index, d, dim_zero_flags_v[i], level_offsets_v[i],
                                  level_packed_v[i], index_packed_v[i]);
    }
  }

  void pack_level_index_dim(uint64_t *level, uint64_t *index, uint64_t d, uint64_t
                            &dim_zero_flags, uint64_t &level_offsets, uint64_t &level_packed,
                            uint64_t &index_packed) {
    dim_zero_flags <<= 1;
    if (level[d] == 1) {
      // dimension marked as level = 1, index = 1
      return;
    }
    // marked dimension as level > 1
    dim_zero_flags += 1;
    uint64_t level_bits = log2(level[d]);
    level_offsets <<= 1;
    level_offsets |= 1;
    level_offsets <<= level_bits;
    level_packed <<= level_bits + 1;
    level_packed |= (level[d] - 2);
    // index is odd, shift out zero
    uint64_t adjusted_index = index[d] >> 1;
    // std::cout << "adjusted_index: " << adjusted_index << std::endl;
    // uint64_t index_bits = (1 << level[d]) - 1;
    // std::cout << "index_bits: " << (level[d] - 1) << std::endl;
    index_packed <<= level[d] - 1;
    index_packed |= adjusted_index;
  }

  __attribute__((noinline)) void get_next(uint64_t &level_d, uint64_t &index_d, uint64_t
                            &dim_zero_flags, uint64_t &level_offsets, uint64_t &level_packed,
                            uint64_t &index_packed) {
    uint64_t is_dim_implicit = dim_zero_flags & one_mask;
    dim_zero_flags >>= 1;
    if (is_dim_implicit == 0) {
      level_d = 1;
      index_d = 1;
      return;
    }
    //uint64_t level_bits = __builtin_ffs(level_offsets);
    // uint64_t level_bits = __builtin_ctz(level_offsets) + 1;
    // uint64_t test1 = static_cast<uint64_t>(1) << 33;
    // uint64_t test2 = static_cast<uint64_t>(1) << 32;
    // uint64_t test3 = static_cast<uint64_t>(1) << 31;
    // uint64_t test4 = static_cast<uint64_t>(1) << 30;
    // uint64_t test5 = level_offsets & static_cast<uint64_t>(-level_offsets);
    // std::cout << sizeof(uint64_t) << std::endl;
    // std::cout << __builtin_clz(test1) << " vs " << __builtin_ctz(level_offsets) + 1 << std::endl;
    // std::cout << __builtin_clz(test2) << " vs " << __builtin_ctz(level_offsets) + 1 << std::endl;
    // std::cout << __builtin_clz(test3) << " vs " << __builtin_ctz(level_offsets) + 1 << std::endl;
    // std::cout << __builtin_clz(test4) << " vs " << __builtin_ctz(level_offsets) + 1 << std::endl;
    uint64_t level_bits = 32 - __builtin_clz(level_offsets & (-level_offsets));
    level_offsets >>= level_bits;
    uint64_t level_mask = (1 << level_bits) - 1;
    level_d = (level_packed & level_mask) + 2;
    level_packed >>= level_bits;
    uint64_t index_bits = level_d - 1;
    uint64_t index_mask = (1 << index_bits) - 1;
    index_d = ((index_packed & index_mask) << 1) + 1;
    index_packed >>= index_bits;
  }

  // TODO: this requries Haswell or newer
  static inline uint64_t log2(const uint64_t x) {
    uint64_t y;
    asm("\tbsr %1, %0\n" : "=r"(y) : "r"(x));
    return y;
  }
  static constexpr uint64_t one_mask = 1;



};
