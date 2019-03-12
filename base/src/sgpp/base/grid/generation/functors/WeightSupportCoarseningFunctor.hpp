#pragma once

#include <sgpp/base/grid/generation/functors/UnlimitedCoarseningFunctor.hpp>

namespace sgpp {
namespace base {
class WeightSupportCoarseningFunctor : public UnlimitedCoarseningFunctor {
protected:
  DataVector &b;
  double threshold;

public:
  WeightSupportCoarseningFunctor(DataVector &b, double threshold = 0.0)
      : b(b), threshold(threshold) {}

  ~WeightSupportCoarseningFunctor() override {}

  bool operator()(GridStorage &storage, size_t seq) override {
    return b[seq] < threshold; // sums of hat functions are positive
  }
};

} // namespace base
} // namespace sgpp
