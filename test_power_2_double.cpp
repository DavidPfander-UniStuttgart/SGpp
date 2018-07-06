#include <cmath>
#include <iostream>

#define FP_BITS 64

constexpr uint64_t DIMS = 20;
constexpr uint64_t BITS_PER_LEVEL = 6; // log_2(64)
constexpr uint64_t DIMS_PER_ELEMENT = FP_BITS / BITS_PER_LEVEL;
constexpr uint64_t ELEMENTS = DIMS % DIMS_PER_ELEMENT == 0
                                  ? DIMS / DIMS_PER_ELEMENT
                                  : DIMS / DIMS_PER_ELEMENT + 1;
constexpr uint64_t LEVEL_MASK = pow(2, BITS_PER_LEVEL) - 1;

double expo_patterns[FP_BITS];

uint64_t index_masks[FP_BITS];

void create_index_masks() {
  for (size_t i = 0; i < FP_BITS; i++) {
    index_masks[i] = (1 << i) - 1;
  }
}

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

void create_expo_patterns() {
  for (int level = 0; level < FP_BITS; level++) {
    expo_patterns[level] = std::pow(2, level);
  }
}

void encode_level(uint64_t level, uint64_t &encoded_levels, uint64_t d) {
  // std::cout << "before encoded: ";
  // print_binary(encoded_levels);
  // std::cout << "level: " << level << std::endl;
  // std::cout << "level bits: ";
  // print_binary(level);
  // encoded_levels <<= BITS_PER_LEVEL;
  // 9 % 5 = 4; 0 % DPE = 0;
  // 5 - 0
  // uint64_t entry_in_element = (DIMS_PER_ELEMENT - 1) - (d %
  // DIMS_PER_ELEMENT);
  level <<= BITS_PER_LEVEL * (d % DIMS_PER_ELEMENT);
  // std::cout << "level bits shifted: ";
  // print_binary(level);
  encoded_levels |= level;
  // std::cout << "after encoded: ";
  // print_binary(encoded_levels);
}

void encode_levels(uint64_t levels[DIMS],
                   uint64_t (&encoded_levels)[ELEMENTS]) {
  uint64_t cur_element = 0;
  for (size_t d = 0; d < DIMS; d++) {
    if (d % DIMS_PER_ELEMENT == 0 && d > 0) {
      cur_element += 1;
    }
    encode_level(levels[d], encoded_levels[cur_element], d);
  }
}

static inline uint64_t log2(const uint64_t x) {
  uint64_t y;
  asm("\tbsr %1, %0\n" : "=r"(y) : "r"(x));
  return y;
}

void encode_index_1d(uint64_t levels[DIMS], uint64_t indices[DIMS],
                     uint64_t &encoded_indices, uint64_t d) {
  // std::cout << "level: " << level //<< " log2(level): " << log2(level)
  //           << " index: " << index << std::endl;
  // print_binary(index);
  // encoded_indices <<= log2(level);
  uint64_t start_bit = 0;
  for (size_t dd = 0; dd < d; dd++) {
    start_bit += levels[dd];
  }
  // std::cout << "start_bit: " << start_bit << std::endl;
  uint64_t encoded_index = indices[d] << start_bit;
  encoded_indices |= encoded_index;
}

void encode_indices(uint64_t levels[DIMS], uint64_t indices[DIMS],
                    uint64_t &encoded_indices) {
  for (uint64_t d = 0; d < DIMS; d++) {
    encode_index_1d(levels, indices, encoded_indices, d);
  }
}

uint64_t get_level(uint64_t (&encoded_levels)[ELEMENTS], uint64_t d) {
  uint64_t element = d / DIMS_PER_ELEMENT; // 9 / 5 = 1
  uint64_t extracted_bits =
      encoded_levels[element] >> (d % DIMS_PER_ELEMENT) * BITS_PER_LEVEL;
  uint64_t masked = extracted_bits & LEVEL_MASK;
  return masked;
}

uint64_t get_index(uint64_t (&encoded_levels)[ELEMENTS],
                   uint64_t encoded_indices, uint64_t d) {
  uint64_t start_bit = 0;
  for (size_t dd = 0; dd < d; dd++) {
    start_bit += get_level(encoded_levels, dd);
  }
  uint64_t level = get_level(encoded_levels, d);
  uint64_t index_mask = (1 << level) - 1;
  uint64_t non_masked = encoded_indices >> start_bit;
  uint64_t index = index_mask & non_masked;
  return index;
}

class packed_grid_point_iterator {
private:
  uint64_t cur_element;
  uint64_t (&encoded_levels)[ELEMENTS];
  uint64_t encoded_indices;
  uint64_t element;

public:
  packed_grid_point_iterator(uint64_t (&encoded_levels)[ELEMENTS],
                             uint64_t &encoded_indices)
      : encoded_levels(encoded_levels), encoded_indices(encoded_indices),
        element(0) {
    cur_element = encoded_levels[0];
  }

  // can only be called DIMS times
  __attribute__((noinline)) void next(uint64_t &level, uint64_t &index) {
    level = cur_element & LEVEL_MASK;
    cur_element >>= BITS_PER_LEVEL;
    index = encoded_indices & index_masks[level];
    encoded_indices >>= level;
    if (cur_element == 0) { // makes use of all levels being >= 1
      element += 1;
      cur_element = encoded_levels[element];
    } // sum: 4 + 1 comp (not counting stuff in if) + 2 convert after return
  }
};

int main() {

  std::cout << "DIMS: " << DIMS << std::endl;
  std::cout << "BITS_PER_LEVEL: " << BITS_PER_LEVEL << std::endl;
  std::cout << "DIMS_PER_ELEMENT: " << DIMS_PER_ELEMENT << std::endl;
  std::cout << "ELEMENTS: " << ELEMENTS << std::endl;

  create_expo_patterns();
  create_index_masks();

  // for (int level = 0; level < FP_BITS; level++) {
  //   double value = std::pow(2, level);
  //   std::cout << "level: " << level << " value: " << value << std::endl;
  //   print_binary(value);
  // }

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
  uint64_t encoded_levels[ELEMENTS] = {0};
  encode_levels(level, encoded_levels);
  std::cout << "level encoded: " << std::endl;
  for (uint64_t e = 0; e < ELEMENTS; e++) {
    std::cout << "e: " << e << " bits: ";
    print_binary(encoded_levels[e], BITS_PER_LEVEL);
  }
  uint64_t encoded_indices = 0;
  encode_indices(level, index, encoded_indices);

  std::cout << "index encoded: " << std::endl;
  print_binary(encoded_indices, level);

  // for (size_t d = 0; d < DIMS; d++) {
  //   std::cout << "level d: " << d << " -> " << get_level(encoded_levels, d)
  //             << std::endl;
  // }

  // std::cout << "index d: " << 0 << " -> "
  //           << get_index(encoded_levels, encoded_indices, 0) << std::endl;

  for (size_t d = 0; d < DIMS; d++) {
    std::cout << "index d: " << d << " -> "
              << get_index(encoded_levels, encoded_indices, d) << std::endl;
  }

  {
    uint64_t l;
    uint64_t i;
    packed_grid_point_iterator it(encoded_levels, encoded_indices);
    for (size_t d = 0; d < DIMS; d++) {
      it.next(l, i);
      std::cout << "l: " << l << " i: " << i << std::endl;
    }
  }

  for (size_t rep = 0; rep < 1000000; rep++) {
    ///{
    uint64_t l;
    uint64_t i;
    packed_grid_point_iterator it(encoded_levels, encoded_indices);
    for (size_t d = 0; d < DIMS; d++) {
      it.next(l, i);
      // std::cout << "l: " << l << " i: " << i << std::endl;
    }
  }

  return 0;
}
