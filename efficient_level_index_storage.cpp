#include <cmath>
#include <iostream>

#define FP_BITS 64

constexpr uint64_t DIMS = 5;
constexpr uint64_t BITS_PER_LEVEL = 6;  // log_2(64)
constexpr uint64_t DIMS_PER_ELEMENT = FP_BITS / BITS_PER_LEVEL;
constexpr uint64_t ELEMENTS =
    DIMS % DIMS_PER_ELEMENT == 0 ? DIMS / DIMS_PER_ELEMENT : DIMS / DIMS_PER_ELEMENT + 1;
constexpr uint64_t LEVEL_MASK = pow(2, BITS_PER_LEVEL) - 1;

void print_binary(double value) {
  uint64_t mask = 0x1;
  uint64_t reint = *reinterpret_cast<uint64_t *>(&value);
  // std::cout << "reint: " << reint << std::endl;
  char bits[FP_BITS];
  for (size_t i = 0; i < FP_BITS; i++) {
    uint64_t anded = mask & reint;
    // std::cout << "anded: " << anded << std::endl;
    if (anded == 1) {
      // std::cout << "1";
      bits[i] = '1';
    } else {
      // std::cout << "0";
      bits[i] = '0';
    }
    reint >>= 1;
  }
  for (int i = FP_BITS - 1; i >= 0; i--) {
    std::cout << bits[i];
  }
  // for (int i = 0; i < FP_BITS; i++) {
  //   std::cout << bits[i];
  // }
  std::cout << std::endl;
}

void print_binary(uint64_t value, uint64_t bounds = 0) {
  uint64_t mask = 0x1;
  // std::cout << "reint: " << reint << std::endl;
  char bits[FP_BITS];
  for (size_t i = 0; i < FP_BITS; i++) {
    uint64_t anded = mask & value;
    // std::cout << "anded: " << anded << std::endl;
    if (anded == 1) {
      // std::cout << "1";
      bits[i] = '1';
    } else {
      // std::cout << "0";
      bits[i] = '0';
    }
    value >>= 1;
  }
  for (int i = FP_BITS - 1; i >= 0; i--) {
    if (bounds > 0 && (i + 1) % bounds == 0) {
      std::cout << "|";
    }
    std::cout << bits[i];
  }
  std::cout << std::endl;
}

void print_binary(uint64_t value, uint64_t (&bounds)[DIMS]) {
  uint64_t mask = 0x1;
  // std::cout << "reint: " << reint << std::endl;
  int32_t bound_index = DIMS - 1;
  int32_t bound_counter = 0;

  uint64_t highest_used_bits = 0;
  for (size_t d = 0; d < DIMS; d++) {
    highest_used_bits += bounds[d];
  }
  uint64_t bits_to_skip = FP_BITS - highest_used_bits;
  uint64_t skip_counter = 0;

  char bits[FP_BITS];
  for (size_t i = 0; i < FP_BITS; i++) {
    uint64_t anded = mask & value;
    if (anded == 1) {
      bits[i] = '1';
    } else {
      bits[i] = '0';
    }
    value >>= 1;
  }
  for (int32_t i = FP_BITS - 1; i >= 0; i--) {
    if (skip_counter == bits_to_skip) {
      if (bound_counter == bounds[bound_index]) {
        std::cout << "|";
        bound_counter = 1;
        bound_index -= 1;
      } else {
        if (bound_index == DIMS - 1 && bound_counter == 0) {
          std::cout << "!";
        }
        bound_counter += 1;
      }
    } else {
      skip_counter += 1;
    }
    std::cout << bits[i];
  }
  std::cout << std::endl;
}

namespace grid {

class li_vector {
 public:
  uint64_t dim_zero_flags;
  uint64_t level_offsets;
  uint64_t level_packed;
  uint64_t index_packed;
  static constexpr uint64_t one_mask = 1;

  // TODO: this requries Haswell or newer
  static inline uint64_t log2(const uint64_t x) {
    uint64_t y;
    asm("\tbsr %1, %0\n" : "=r"(y) : "r"(x));
    return y;
  }

 private:
  void pack_level_index_dim(uint64_t (&level)[DIMS], uint64_t (&index)[DIMS], uint64_t d) {
    std::cout << "------------------------------------" << std::endl;
    std::cout << "cur l: " << level[d] << " i: " << index[d] << std::endl;
    // std::cout << "dim_zero_flags: " << std::endl;
    // print_binary(dim_zero_flags);
    dim_zero_flags <<= 1;
    if (level[d] == 1) {
      // dimension marked as level = 1, index = 1
      std::cout << "empty" << std::endl;
      return;
    }
    std::cout << "l - 2: " << (level[d] - 2) << std::endl;
    // marked dimension as level > 1
    dim_zero_flags += 1;
    // print_binary(dim_zero_flags);
    uint64_t level_bits = log2(level[d]);
    std::cout << "level_bits: " << level_bits << std::endl;
    level_offsets <<= level_bits;
    level_offsets |= level_bits;
    level_packed <<= level_bits;
    level_packed |= (level[d] - 2);
    // index is odd, shift out zero
    uint64_t adjusted_index = index[d] >> 1;
    std::cout << "adjusted_index: " << adjusted_index << std::endl;
    // uint64_t index_bits = (1 << level[d]) - 1;
    std::cout << "index_bits: " << (level[d] - 1) << std::endl;
    index_packed <<= level[d] - 1;
    index_packed |= adjusted_index;
  }

 public:
  li_vector(uint64_t (&level)[DIMS], uint64_t (&index)[DIMS])
      : dim_zero_flags(0), level_offsets(0), level_packed(0), index_packed(0) {
    for (uint64_t d = 0; d < DIMS; d++) {
      pack_level_index_dim(level, index, d);
    }
    std::cout << "dim_zero_flags" << std::endl;
    print_binary(dim_zero_flags);
    std::cout << "level_offsets" << std::endl;
    print_binary(level_offsets);
    std::cout << "level_packed" << std::endl;
    print_binary(level_packed);
    std::cout << "index_packed" << std::endl;
    print_binary(index_packed);
  }

  void get_next(uint64_t &level_d, uint64_t &index_d) {
    uint64_t is_dim_implicit = dim_zero_flags & one_mask;
    if (is_dim_implicit == 0) {
      level_d = 1;
      index_d = 1;
      return;
    }
    dim_zero_flags >>= 1;
    uint64_t level_bits = __builtin_ffs(level_offsets);
    uint64_t level_mask = (1 << level_bits) - 1;
    level_d = (level_packed & level_mask) + 2;
    level_mask >>= level_bits;
    uint64_t index_bits = 1 << level_d;
    uint64_t index_mask = index_bits - 1;
    index_d = ((index_packed & index_mask) << 1) + 1;
    index_packed >>= index_bits;
  }
};
}

int main() {
  std::cout << "DIMS: " << DIMS << std::endl;
  std::cout << "BITS_PER_LEVEL: " << BITS_PER_LEVEL << std::endl;
  std::cout << "DIMS_PER_ELEMENT: " << DIMS_PER_ELEMENT << std::endl;
  std::cout << "ELEMENTS: " << ELEMENTS << std::endl;

  uint64_t level[DIMS];
  for (size_t d = 0; d < DIMS; d++) {
    level[d] = (d % 5) + 1;
  }
  uint64_t index[DIMS];
  for (size_t d = 0; d < DIMS; d++) {
    if (d % 2 == 0) {
      index[d] = std::pow(2, level[d]) - 1;
    } else {
      index[d] = 3;
    }
  }
  {
    for (size_t d = 0; d < DIMS; d++) {
      std::cout << "original d: " << d << " l: " << level[d] << " i: " << index[d] << std::endl;
    }
  }

  {
    grid::li_vector li(level, index);
    uint64_t l;
    uint64_t i;
    for (size_t d = 0; d < DIMS; d++) {
      li.get_next(l, i);
      std::cout << "l: " << l << " i: " << i << std::endl;
    }
  }

  return 0;
}
