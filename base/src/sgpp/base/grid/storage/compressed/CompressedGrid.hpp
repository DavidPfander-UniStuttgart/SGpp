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

template<class T = uint64_t>
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
  template<class G>
  compressed_grid(std::vector<G> &level, std::vector<G> &index, size_t dimensions) :
      gridsize(level.size()/(dimensions)), dims(dimensions),
      dim_zero_flags_v(level.size()/(dimensions)),
      level_offsets_v(level.size()/(dimensions), 0),
      level_packed_v(level.size()/(dimensions), 0),
      index_packed_v(level.size()/(dimensions), 0)
  {
    for (size_t i = 0; i < gridsize; i++) {
      pack_gridpoint(level, index, i);
    }
  }

 public:
  std::vector<T> dim_zero_flags_v;
  std::vector<T> level_offsets_v;
  std::vector<T> level_packed_v;
  std::vector<T> index_packed_v;

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
      T dim_zero_flags_copy = dim_zero_flags_v[i];
      T level_offsets_copy = level_offsets_v[i];
      T level_packed_copy = level_packed_v[i];
      T index_packed_copy = index_packed_v[i];
      for (size_t d = 0; d < dims; d++) {
        T extracted_level = 0;
        T extracted_index = 0;
        get_next(extracted_level, extracted_index, dim_zero_flags_copy, level_offsets_copy,
                 level_packed_copy, index_packed_copy);
        const int grid_index = gridpoints[i * 2 * dims + 2 * d];
        const int grid_level = gridpoints[i * 2 * dims + 2 * d + 1];
        if (grid_index != extracted_index || grid_level != extracted_level) {
          if (grid_index == 0 && grid_level == 0) // padding point (0,0) - we cannot compress that
            //anyway - in this case the wrong result is expected. This needs to be handled in the opencl kernels
            continue;
          std::cout << extracted_index << " vs " << grid_index << std::endl;
          std::cout << extracted_level << " vs " << grid_level << std::endl;
          std::cin.get();
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
    T level[dims];
    T index[dims];
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

  template<class G>
  void pack_gridpoint(std::vector<G> &levels, std::vector<G> &indices, size_t i) {
    T level[dims];
    T index[dims];
    // stage values
    for (int64_t d = 0; d < dims; d++) {
      index[d] = static_cast<T>(indices[i * dims + d]);
      level[d] = static_cast<T>(levels[i * dims + d]);
    }
    // pack values into the compressed vectors
    for (int64_t d = dims - 1; d >= 0; d--) {
      pack_level_index_dim(level, index, d, dim_zero_flags_v[i], level_offsets_v[i],
                                  level_packed_v[i], index_packed_v[i]);
    }
  }

  void pack_level_index_dim(T *level, T *index, T d, T
                            &dim_zero_flags, T &level_offsets, T &level_packed,
                            T &index_packed) {
    dim_zero_flags <<= 1;
    if (level[d] == 1) {
      // dimension marked as level = 1, index = 1
      return;
    }
    // marked dimension as level > 1
    dim_zero_flags += 1;
    T level_bits = log2(static_cast<uint64_t>(level[d]));
    level_offsets >>= 1;
    if(std::is_same<T, uint64_t>::value)
      level_offsets |= (static_cast<T>(1)<< 63);
    if(std::is_same<T, unsigned int>::value)
      level_offsets |= (static_cast<T>(1)<< 31);
    if(std::is_same<T, int>::value)
      level_offsets |= (static_cast<T>(1)<< 31);
    level_offsets >>= level_bits;
    level_packed <<= level_bits + 1;
    level_packed |= (level[d] - 2);
    // index is odd, shift out zero
    T adjusted_index = index[d] >> 1;
    index_packed <<= level[d] - 1;
    index_packed |= adjusted_index;
  }

  __attribute__((noinline)) void get_next(T &level_d, T &index_d, T
                            &dim_zero_flags, T &level_offsets, T &level_packed,
                            T &index_packed) {
    T is_dim_implicit = dim_zero_flags & one_mask;
    dim_zero_flags >>= 1;
    if (is_dim_implicit == 0) {
      level_d = 1;
      index_d = 1;
      return;
    }
    T level_bits;
    if(std::is_same<T, uint64_t>::value)
      level_bits = __builtin_clzll(level_offsets) + 1;
    if(std::is_same<T, unsigned int>::value)
      level_bits = __builtin_clz(level_offsets) + 1;
    if(std::is_same<T, int>::value)
      level_bits = __builtin_clz(level_offsets) + 1;
    level_offsets <<= level_bits;
    T level_mask = (1 << level_bits) - 1;
    level_d = (level_packed & level_mask) + 2;
    level_packed >>= level_bits;
    T index_bits = level_d - 1;
    T index_mask = (1 << index_bits) - 1;
    index_d = ((index_packed & index_mask) << 1) + 1;
    index_packed >>= index_bits;
  }

  // TODO: this requries Haswell or newer
  static inline T log2(const uint64_t x) {
    uint64_t y;
    asm("\tbsr %1, %0\n" : "=r"(y) : "r"(x));
    return static_cast<T>(y);
  }
  static constexpr T one_mask = 1;



};
