// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#ifndef UNLIMITEDCOARSENINGFUNCTOR_HPP
#define UNLIMITEDCOARSENINGFUNCTOR_HPP

#include <sgpp/base/grid/GridStorage.hpp>

#include <sgpp/globaldef.hpp>


namespace sgpp {
namespace base {

/**
 * Abstract class that defines the interfaces that coarsening functors have to provide.
 */
class UnlimitedCoarseningFunctor {
 public:

  /**
   * Constructor
   */
  UnlimitedCoarseningFunctor() {}

  /**
   * Destructor
   */
  virtual ~UnlimitedCoarseningFunctor() {}

  /**
   * Return true if grid points shall be removed
   *
   * @param storage reference to the grids storage object
   * @param seq sequence number in the coefficients array
   *
   * @return refinement value
   */
  virtual bool operator()(GridStorage& storage, size_t seq) = 0;
};

}  // namespace base
}  // namespace sgpp

#endif /* UNLIMITEDCOARSENINGFUNCTOR_HPP */
