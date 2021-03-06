// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

// include all combigrid headers
#include <sgpp/combigrid/operation/CombigridMultiOperation.hpp>
#include <sgpp/combigrid/operation/CombigridOperation.hpp>
#include <sgpp/combigrid/operation/Configurations.hpp>
#include <sgpp/combigrid/operation/multidim/AveragingLevelManager.hpp>
#include <sgpp/combigrid/operation/multidim/WeightedRatioLevelManager.hpp>
#include <sgpp/combigrid/storage/FunctionLookupTable.hpp>
#include <sgpp/combigrid/storage/tree/CombigridTreeStorage.hpp>
#include <sgpp/combigrid/utils/Stopwatch.hpp>
#include <sgpp/combigrid/utils/Utils.hpp>
#include <sgpp_combigrid.hpp>

#include <sgpp/base/operation/hash/common/basis/BsplineBasis.hpp>
#include <sgpp/combigrid/operation/onedim/BSplineInterpolationEvaluator.hpp>

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using sgpp::base::DataVector;
using sgpp::combigrid::MultiFunction;
using sgpp::combigrid::CombigridOperation;
using sgpp::combigrid::AbstractPointHierarchy;
using sgpp::combigrid::NestedPointHierarchy;
using sgpp::combigrid::LejaPointDistribution;
using sgpp::combigrid::IdentityPointOrdering;
using sgpp::combigrid::LinearGrowthStrategy;
using sgpp::combigrid::AbstractLinearEvaluator;
using sgpp::combigrid::FloatArrayVector;
using sgpp::combigrid::FloatScalarVector;
using sgpp::combigrid::ArrayEvaluator;
using sgpp::combigrid::PolynomialInterpolationEvaluator;
using sgpp::combigrid::CombigridTreeStorage;
using sgpp::combigrid::FunctionLookupTable;
using sgpp::combigrid::CombigridEvaluator;
using sgpp::combigrid::WeightedRatioLevelManager;

// Plotting B-Spline basis functions

int main() {
  size_t d = 1;
  size_t degree = 5;
  size_t level = 5;
  //  std::vector<double> GridPoints = { 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875};

  sgpp::combigrid::CombiHierarchies::Collection grids(
      d, sgpp::combigrid::CombiHierarchies::expUniformBoundary());

  //  sgpp::combigrid::FloatScalarVector EvalPoint(0.5);
  sgpp::combigrid::FloatArrayVector EvalPoints;
  //  EvalPoint[0].value() = 0.25;
  //  EvalPoint[1].value() = 0.5;
  //  EvalPoint[2].value() = 0.75;

  // uniform grid
  double gridwidth = 0.001;
  for (size_t i = 0; i < 1 / gridwidth + 1; i++) {
    EvalPoints[i] = static_cast<double>(i) * gridwidth;
  }

  std::string plotstr = "/home/rehmemk/SGS_Sync/Plotting/combigrid_bsplines/bsplines.dat";
  remove(plotstr.c_str());
  std::ofstream plotfile;
  plotfile.open(plotstr.c_str(), std::ios::app);
  plotfile << "#Basis functions  \n";

  sgpp::combigrid::CombiEvaluators::MultiCollection evaluators(
      d, sgpp::combigrid::CombiEvaluators::multiBSplineInterpolation(degree));
  //  sgpp::combigrid::CombiEvaluators::MultiCollection evaluators(
  //      d, sgpp::combigrid::CombiEvaluators::multiLinearInterpolation());

  sgpp::combigrid::CombiEvaluators::MultiCollection evalCopy(d);
  for (size_t dim = 0; dim < d; ++dim) {
    evalCopy[dim] = evaluators[dim]->cloneLinear();
    bool needsSorted = true;

    auto GridPoints = grids[dim]->getPoints(level, needsSorted);
    for (size_t i = 0; i < GridPoints.size(); i++) {
      std::cout << GridPoints[i] << " ";
    }
    std::cout << "\n";
    evalCopy[dim]->setGridPoints(GridPoints);
    evalCopy[dim]->setParameter(EvalPoints);
    std::vector<FloatArrayVector> basisValues1D = evalCopy[dim]->getBasisValues();

    for (size_t i = 0; i < EvalPoints.size(); i++) {
      plotfile << EvalPoints[i].value() << ", ";
      for (size_t j = 0; j < GridPoints.size() - 1; j++) {
        plotfile << basisValues1D[j][i].value() << ", ";
      }
      plotfile << basisValues1D[GridPoints.size() - 1][i].value() << "\n";
    }
  }
  plotfile.close();
}
