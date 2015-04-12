// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#include <sgpp/base/grid/GridStorage.hpp>

#include <sgpp/base/grid/generation/hashmap/HashGenerator.hpp>

#include <sgpp/globaldef.hpp>
#include "GeneralizedTruncatedBoundaryGridGenerator.hpp"


namespace SGPP {
  namespace base {

    GeneralizedTruncatedBoundaryGridGenerator::GeneralizedTruncatedBoundaryGridGenerator(GridStorage* storage) : storage(storage) {
    }

    GeneralizedTruncatedBoundaryGridGenerator::~GeneralizedTruncatedBoundaryGridGenerator() {
    }

    void GeneralizedTruncatedBoundaryGridGenerator::regular(size_t level) {
      this->truncated( static_cast<HashGenerator::level_t>(level), 1);
    }

    void GeneralizedTruncatedBoundaryGridGenerator::cliques(size_t level, size_t clique_size) {
      throw generation_exception("Method is not implemented");
    }


    void GeneralizedTruncatedBoundaryGridGenerator::truncated(size_t level, size_t l_user) {
      HashGenerator gen;
      gen.truncated(this->storage, static_cast<HashGenerator::level_t>(level), static_cast<HashGenerator::level_t>(l_user));
    }

    //void BoundaryGridGenerator::refine(RefinementFunctor* func)
    //{
    //  HashRefinementBoundaries refine;
    //  refine.free_refine(this->storage, func);
    //}
    //
    //size_t BoundaryGridGenerator::getNumberOfRefinablePoints()
    //{
    //  HashRefinementBoundaries refine;
    //  return refine.getNumberOfRefinablePoints(this->storage);
    //}
    //
    //void BoundaryGridGenerator::coarsen(CoarseningFunctor* func, DataVector* alpha)
    //{
    //  HashCoarsening coarsen;
    //  coarsen.free_coarsen(this->storage, func, alpha);
    //}
    //
    //size_t BoundaryGridGenerator::getNumberOfRemovablePoints()
    //{
    //  HashCoarsening coarsen;
    //  return coarsen.getNumberOfRemovablePoints(this->storage);
    //}
    //
    //void BoundaryGridGenerator::refineMaxLevel(RefinementFunctor* func, unsigned int maxLevel)
    //{
    //  HashRefinementBoundariesMaxLevel refine;
    //  refine.refineToMaxLevel(this->storage, func, maxLevel);
    //}
    //
    //size_t BoundaryGridGenerator::getNumberOfRefinablePointsToMaxLevel(unsigned int maxLevel)
    //{
    //  HashRefinementBoundariesMaxLevel refine;
    //  return refine.getNumberOfRefinablePointsToMaxLevel(this->storage, maxLevel);
    //}

  }
}
