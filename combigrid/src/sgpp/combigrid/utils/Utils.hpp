// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <sgpp/globaldef.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace sgpp {
namespace combigrid {

template <typename T>
T pow(T base, size_t exponent) {
  T result = 1;

  size_t mask = static_cast<size_t>(1) << (8 * sizeof(size_t) - 1);

  while (exponent != 0) {
    result *= result;

    if (exponent & mask) {
      result *= base;
    }

    exponent = exponent << 1;
  }

  return result;
}

std::int64_t binom(std::int64_t n, std::int64_t k);

std::vector<std::string> split(std::string str, std::string separator);
std::string join(std::vector<std::string> const &elements, std::string const &separator);
std::string escape(std::string str, char escapeCharacter, std::string avoidCharacters,
                   std::string replaceCharacters);
std::string unescape(std::string str, char escapeCharacter, std::string avoidCharacters,
                     std::string replaceCharacters);
}  // namespace combigrid
}  // namespace sgpp

#endif /* UTILS_HPP_ */