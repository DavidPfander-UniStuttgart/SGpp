// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org
#pragma once

#include "sgpp/base/grid/Grid.hpp"
#include "sgpp/base/grid/GridStorage.hpp"
#include "sgpp/globaldef.hpp"

namespace sgpp {
namespace datadriven {

// Provides a-priori refinement that guesses a grid for a given dataset.
// Goal is to avoid some of the solver steps as classical refinement only
// refines a single level and then again solves the system of linear equations.
class OperationDatadrivenRefinement {
 public:

  void refine(sgpp::base::Grid &grid, sgpp::base::DataMatrix &data,
              sgpp::base::DataVector &values) {
      // refine_(grid, data, values);
  }

 private:

  // void refine_(sgpp::base::Grid &grid, sgpp::base::DataMatrix &data,
  //             sgpp::base::DataVector &values) {
  //     auto& storage = grid->getStorage();
  // }
};

}  // namespace datadriven
}  // namespace sgpp
