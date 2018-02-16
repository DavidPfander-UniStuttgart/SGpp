// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#include <sgpp/combigrid/algebraic/FloatScalarVector.hpp>
#include <sgpp/combigrid/operation/CombigridMultiOperation.hpp>
#include <sgpp/combigrid/operation/CombigridOperation.hpp>
#include <sgpp/combigrid/operation/Configurations.hpp>
#include <sgpp/combigrid/operation/multidim/AveragingLevelManager.hpp>
#include <sgpp/combigrid/operation/multidim/WeightedRatioLevelManager.hpp>
#include <sgpp/combigrid/operation/onedim/BSplineQuadratureEvaluator.hpp>
#include <sgpp/combigrid/operation/onedim/QuadratureEvaluator.hpp>
#include <sgpp/combigrid/storage/FunctionLookupTable.hpp>
#include <sgpp/combigrid/storage/tree/CombigridTreeStorage.hpp>
#include <sgpp/combigrid/utils/Stopwatch.hpp>
#include <sgpp/combigrid/utils/Utils.hpp>

#include <sgpp/optimization/function/scalar/InterpolantScalarFunction.hpp>
#include <sgpp/optimization/sle/solver/Auto.hpp>
#include <sgpp/optimization/sle/system/FullSLE.hpp>
#include <sgpp/optimization/tools/Printer.hpp>
#include <sgpp/quadrature/sampling/NaiveSampleGenerator.hpp>

#include <sgpp/combigrid/operation/CombigridOperation.hpp>
#include <sgpp/combigrid/operation/Configurations.hpp>
#include <sgpp/combigrid/operation/multidim/CombigridEvaluator.hpp>
#include <sgpp/combigrid/operation/multidim/WeightedRatioLevelManager.hpp>
#include <sgpp/combigrid/operation/onedim/BSplineMixedQuadratureEvaluator.hpp>
#include <sgpp/combigrid/storage/tree/CombigridTreeStorage.hpp>
#include <sgpp/optimization/sle/solver/Auto.hpp>
#include <sgpp/optimization/sle/system/FullSLE.hpp>
#include <sgpp/optimization/tools/Printer.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

double f(sgpp::base::DataVector const& v) {
  return 1;
  //  return v[0] * sin(v[0] + v[1]) * exp(v[1] * v[2]);
  //  return v[0] * sin(v[1]) ;
  //  return std::atan(50 * (v[0] - .35)) + M_PI / 2 + 4 * std::pow(v[1], 3) +
  //         std::exp(v[0] * v[1] - 1);
}

void interpolate(size_t maxlevel, size_t numDimensions, size_t degree, double& max_err,
                 double& L2_err) {
  sgpp::combigrid::MultiFunction func(f);
  auto operation =
      sgpp::combigrid::CombigridOperation::createExpUniformBoundaryBsplineInterpolation(
          numDimensions, func, degree);

  //  auto operation =
  //      sgpp::combigrid::CombigridOperation::createExpClenshawCurtisPolynomialInterpolation(
  //          numDimensions, func);

  //  auto operation =
  //  sgpp::combigrid::CombigridOperation::createExpUniformBoundaryLinearInterpolation(
  //      numDimensions, func);

  double diff = 0.0;
  // generator generates num_points random points in [0,1]^numDimensions
  size_t num_points = 1000;
  sgpp::quadrature::NaiveSampleGenerator generator(numDimensions);
  sgpp::base::DataVector p(numDimensions, 0);

  for (size_t i = 0; i < num_points; i++) {
    generator.getSample(p);
    diff = fabs(operation->evaluate(maxlevel, p) - f(p));
    max_err = (diff > max_err) ? diff : max_err;
    L2_err += pow(diff, 2);
  }
  L2_err = sqrt(L2_err / static_cast<double>(num_points));

  std::cout << "# grid points: " << operation->numGridPoints() << " ";
}

/**
 * @param level level of the underlying 1D subspace
 * @return vector containing the integrals of all basisfunctions
 */
std::vector<double> integrateBasisFunctions(size_t level, size_t numDimensions, size_t degree) {
  sgpp::combigrid::CombiHierarchies::Collection grids(
      numDimensions, sgpp::combigrid::CombiHierarchies::expUniformBoundary());
  bool needsOrdered = true;

  auto evaluator = sgpp::combigrid::CombiEvaluators::BSplineQuadrature(degree);
  evaluator->setGridPoints(grids[0]->getPoints(level, needsOrdered));
  std::vector<sgpp::combigrid::FloatScalarVector> weights = evaluator->getBasisValues();
  std::vector<double> integrals(weights.size());
  for (size_t i = 0; i < weights.size(); i++) {
    integrals[i] = weights[i].value();
  }
  return integrals;
}

double integrate(size_t level, size_t numDimensions, size_t degree) {
  sgpp::combigrid::MultiFunction func(f);
  auto operation = sgpp::combigrid::CombigridOperation::createExpUniformBoundaryBsplineQuadrature(
      numDimensions, func, degree);
  return operation->evaluate(level);
}

/**
   * @param numDimensions   number of dimensions
   * @param func     		objective function
   * @param degree       	B-spline degree
   * * @return      		Operation for calculating mean(u^2) where u is the B-Spline
 * interpolant
 * of
   * 						func
   * Mean(u^2) = \int u^2(x) f(x) dx = \sum_i \alpha_i \sum_j \alpha_j \int b_i b_j dx
   * Therefore this operation uses the special quadrature for \int b_i b_j and a custom grid
 * function that calculated the interpolation coefficients \alpha and saves the products \alpha_i
 * \alpha_j into a TreeStorage
   */
double BsplineMeanSquare(size_t level, size_t numDimensions, sgpp::combigrid::MultiFunction func,
                         size_t degree) {
  sgpp::combigrid::CombiHierarchies::Collection grids(
      numDimensions, sgpp::combigrid::CombiHierarchies::expUniformBoundary());

  sgpp::combigrid::CombiEvaluators::Collection evaluators(
      numDimensions, sgpp::combigrid::CombiEvaluators::BSplineMixedQuadrature(degree));

  // So far only WeightedRatioLevelManager has been used
  std::shared_ptr<sgpp::combigrid::LevelManager> levelManager(
      new sgpp::combigrid::WeightedRatioLevelManager());

  // stores the values of the objective function
  auto funcStorage = std::make_shared<sgpp::combigrid::CombigridTreeStorage>(grids, func);

  // Grid Function that calculates the coefficients for the B-Spline interpolation.
  // The coeficients for each B-Spline are saved in a TreeStorage encoded by a MultiIndex
  sgpp::combigrid::GridFunction gf([=](std::shared_ptr<sgpp::combigrid::TensorGrid> grid) {
    sgpp::combigrid::CombiEvaluators::Collection interpolEvaluators(
        numDimensions, sgpp::combigrid::CombiEvaluators::BSplineInterpolation(degree));
    size_t numDimensions = grid->getDimension();
    auto coefficientTree = std::make_shared<sgpp::combigrid::TreeStorage<double>>(numDimensions);
    auto level = grid->getLevel();
    std::vector<size_t> numGridPointsVec = grid->numPoints();
    size_t numGridPoints = 1;
    for (size_t i = 0; i < numGridPointsVec.size(); i++) {
      numGridPoints *= numGridPointsVec[i];
    }

    sgpp::combigrid::CombiEvaluators::Collection evalCopy(numDimensions);
    for (size_t dim = 0; dim < numDimensions; ++dim) {
      evalCopy[dim] = interpolEvaluators[dim]->cloneLinear();
      bool needsSorted = evalCopy[dim]->needsOrderedPoints();
      auto gridPoints = grids[dim]->getPoints(level[dim], needsSorted);
      evalCopy[dim]->setGridPoints(gridPoints);
    }
    sgpp::base::DataMatrix A(numGridPoints, numGridPoints);
    sgpp::base::DataVector coefficients_sle(numGridPoints);
    sgpp::base::DataVector functionValues(numGridPoints);

    // Creates an iterator that yields the multi-indices of all grid points in the grid.
    sgpp::combigrid::MultiIndexIterator it(grid->numPoints());
    auto funcIter =
        funcStorage->getGuidedIterator(level, it, std::vector<bool>(numDimensions, true));

    for (size_t ixEvalPoints = 0; funcIter->isValid(); ++ixEvalPoints, funcIter->moveToNext()) {
      auto gridPoint = grid->getGridPoint(funcIter->getMultiIndex());
      functionValues[ixEvalPoints] = funcIter->value();

      std::vector<std::vector<double>> basisValues;
      for (size_t dim = 0; dim < numDimensions; ++dim) {
        evalCopy[dim]->setParameter(sgpp::combigrid::FloatScalarVector(gridPoint[dim]));
        auto basisValues1D = evalCopy[dim]->getBasisValues();
        // basis values at gridPoint
        std::vector<double> basisValues1D_vec(basisValues1D.size());
        for (size_t i = 0; i < basisValues1D.size(); i++) {
          basisValues1D_vec[i] = basisValues1D[i].value();
        }
        basisValues.push_back(basisValues1D_vec);
      }

      sgpp::combigrid::MultiIndexIterator innerIter(grid->numPoints());
      for (size_t ixBasisFunctions = 0; innerIter.isValid();
           ++ixBasisFunctions, innerIter.moveToNext()) {
        double splineValue = 1.0;
        auto innerIndex = innerIter.getMultiIndex();
        for (size_t dim = 0; dim < numDimensions; ++dim) {
          splineValue *= basisValues[dim][innerIndex[dim]];
        }
        A.set(ixEvalPoints, ixBasisFunctions, splineValue);
      }
    }

    sgpp::optimization::FullSLE sle(A);
    sgpp::optimization::sle_solver::Auto solver;
    sgpp::optimization::Printer::getInstance().setVerbosity(-1);
    bool solved = solver.solve(sle, functionValues, coefficients_sle);

    if (!solved) {
      exit(-1);
    }

    it.reset();

    // Write coefficients_i * coefficients_j into coefficientTree
    std::cout << "coeff indices:\n";
    for (size_t vecIndex_i = 0; it.isValid(); ++vecIndex_i, it.moveToNext()) {
      //      for (size_t vecIndex_j = 0; it.isValid(); ++vecIndex_j, it.moveToNext()) {
      //        coefficientTree->set(it.getMultiIndex(),
      //                             coefficients_sle[vecIndex_i] * coefficients_sle[vecIndex_j]);
      //        std::cout << it.getMultiIndex().size() << std::endl;
      //        std::cout << " " << vecIndex_i << " " << vecIndex_j << " " <<
      //        it.getMultiIndex()[0]
      //                  << " | ";
      for (size_t bjo = 0; bjo < it.getMultiIndex().size(); bjo++) {
        std::cout << it.getMultiIndex()[bjo] << " ";
      }
      std::cout << "\n";
      //      }
    }
    std::cout << "\n";

    return coefficientTree;
  });

  bool exploitNesting = false;
  auto operation = std::make_shared<sgpp::combigrid::CombigridOperation>(
      grids, evaluators, levelManager, gf, exploitNesting);

  sgpp::base::DataVector p(numDimensions, 0.5);
  double res = operation->evaluate(level, p);
  return res;
}

int main() {
  size_t numDimensions = 1;
  size_t degree = 3;

  // Interpolation
  sgpp::base::SGppStopwatch watch;
  watch.start();
  size_t minLevel = 0;
  size_t maxLevel = 8;

  //  std::vector<double> maxErr(maxLevel + 1, 0);
  //  std::vector<double> L2Err(maxLevel + 1, 0);
  //  for (size_t l = minLevel; l < maxLevel + 1; l++) {
  //    interpolate(l, numDimensions, degree, maxErr[l], L2Err[l]);
  //    std::cout << "level: " << l << " max err " << maxErr[l] << " L2 err " << L2Err[l] <<
  //    std::endl;
  //  }
  //  std::cout << " Total Runtime: " << watch.stop() << " s" << std::endl;

  // Integration
  //  size_t level = 3;
  //  double integral = integrate(level, numDimensions, degree);
  //  std::cout << "integral:  " << integral << std::endl;

  // Integrate basis functions
  //  size_t level = 1;
  //  std::vector<double> integrals = integrateBasisFunctions(level, numDimensions, degree);
  //  std::cout << "------------------------------------" << std::endl;
  //  for (size_t i = 0; i < integrals.size(); i++) {
  //    std::cout << integrals[i] << " ";
  //  }

  return 0;
}