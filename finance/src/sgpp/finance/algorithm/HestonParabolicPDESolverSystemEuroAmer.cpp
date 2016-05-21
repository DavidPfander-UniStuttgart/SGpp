// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#include <sgpp/finance/algorithm/HestonParabolicPDESolverSystemEuroAmer.hpp>
#include <sgpp/base/exception/algorithm_exception.hpp>
#include <sgpp/base/exception/application_exception.hpp>
#include <sgpp/base/grid/generation/functors/SurplusCoarseningFunctor.hpp>
#include <sgpp/base/grid/generation/functors/SurplusRefinementFunctor.hpp>
#include <sgpp/base/operation/BaseOpFactory.hpp>
#include <sgpp/pde/operation/PdeOpFactory.hpp>
#include <sgpp/finance/operation/FinanceOpFactory.hpp>

// #define NOBOUNDARYDISCOUNT

#include <sgpp/globaldef.hpp>

#include <cmath>
#include <string>
#include <vector>

namespace sgpp {
namespace finance {

HestonParabolicPDESolverSystemEuroAmer::HestonParabolicPDESolverSystemEuroAmer(
    sgpp::base::Grid& SparseGrid, sgpp::base::DataVector& alpha, sgpp::base::DataVector& thetas,
    sgpp::base::DataVector& volvols, sgpp::base::DataVector& kappas, sgpp::base::DataMatrix& rho,
    double r, double TimestepSize, std::string OperationMode, double dStrike,
    std::string option_type, bool bLogTransform, bool useCoarsen, double coarsenThreshold,
    std::string adaptSolveMode, int numCoarsenPoints, double refineThreshold,
    std::string refineMode, sgpp::base::GridIndex::level_type refineMaxLevel) {
  this->BoundGrid = &SparseGrid;
  this->alpha_complete = &alpha;

  this->alpha_complete_old = new sgpp::base::DataVector(*this->alpha_complete);
  this->alpha_complete_tmp = new sgpp::base::DataVector(*this->alpha_complete);
  this->oldGridStorage = new sgpp::base::GridStorage(this->BoundGrid->getStorage());
  this->secondGridStorage = new sgpp::base::GridStorage(this->BoundGrid->getStorage());

  this->InnerGrid = NULL;
  this->alpha_inner = NULL;
  this->tOperationMode = OperationMode;
  this->TimestepSize = TimestepSize;
  this->TimestepSize_old = TimestepSize;
  this->BoundaryUpdate = new sgpp::base::DirichletUpdateVector(SparseGrid.getStorage());
  this->GridConverter = new sgpp::base::DirichletGridConverter();
  this->r = r;
  this->thetas = &thetas;
  this->volvols = &volvols;
  this->kappas = &kappas;
  this->hMatrix = &rho;
  this->HestonAlgoDims = this->BoundGrid->getAlgorithmicDimensions();
  this->nAssets = this->HestonAlgoDims.size() / 2;
  this->nExecTimesteps = 0;

  // throw an exception if we're not using Log coordinates and there is more than one asset
  if (this->nAssets > 1 && !bLogTransform) {
    throw sgpp::base::application_exception(
        "HestonParabolicPDESolverSystemEuropean::HestonParabolicPDESolverSystemEuropean : "
        "Cartesian coordinates are not supported for the basket case.");
  }

  // throw exception if algorithmic dimensions isn't even
  if ((this->HestonAlgoDims.size() % 2) != 0) {
    throw sgpp::base::algorithm_exception(
        "HestonParabolicPDESolverSystemEuropean::HestonParabolicPDESolverSystemEuropean : Number "
        "of algorithmic dimensions is not even.");
  }

  // throw exception if grid dimensions not equal algorithmic dimensions
  if (this->HestonAlgoDims.size() != this->BoundGrid->getDimension()) {
    throw sgpp::base::algorithm_exception(
        "HestonParabolicPDESolverSystemEuropean::HestonParabolicPDESolverSystemEuropean : Number "
        "of algorithmic dimensions is not equal to the number of grid's dimensions.");
  }

  // Test if 2*dimC = dimG, where dimC is the number of dimensions in the coefficient vectors and
  // dimG is the number of grid dimensions
  if (this->BoundGrid->getDimension() != (2 * this->thetas->getSize()) ||
      this->BoundGrid->getDimension() != (2 * this->kappas->getSize()) ||
      this->BoundGrid->getDimension() != (2 * this->volvols->getSize())) {
    throw sgpp::base::algorithm_exception(
        "HestonParabolicPDESolverSystemEuropean::HestonParabolicPDESolverSystemEuropean : "
        "Dimension of theta/volvol/kappa parameters != half of grid's dimensions!");
  }

  // Test if number of dimensions in the coefficients match the numbers of grid dimensions (hmatrix)
  if (this->BoundGrid->getDimension() != this->hMatrix->getNrows() ||
      this->BoundGrid->getDimension() != this->hMatrix->getNcols()) {
    throw sgpp::base::algorithm_exception(
        "HestonParabolicPDESolverSystemEuropean::HestonParabolicPDESolverSystemEuropean : Row or "
        "col of hmatrix parameter don't match the grid's dimensions!");
  }

  // Test if all algorithmic dimensions are inside the grid's dimensions
  for (size_t i = 0; i < this->HestonAlgoDims.size(); i++) {
    if (this->HestonAlgoDims[i] >= this->BoundGrid->getDimension()) {
      throw sgpp::base::algorithm_exception(
          "HestonParabolicPDESolverSystemEuropean::HestonParabolicPDESolverSystemEuropean : "
          "Minimum one algorithmic dimension is not inside the grid's dimensions!");
    }
  }

  // Test if there are double algorithmic dimensions
  std::vector<size_t> tempAlgoDims(this->HestonAlgoDims);

  for (size_t i = 0; i < this->HestonAlgoDims.size(); i++) {
    size_t dimCount = 0;

    for (size_t j = 0; j < tempAlgoDims.size(); j++) {
      if (this->HestonAlgoDims[i] == tempAlgoDims[j]) {
        dimCount++;
      }
    }

    if (dimCount > 1) {
      throw sgpp::base::algorithm_exception(
          "HestonParabolicPDESolverSystemEuropean::HestonParabolicPDESolverSystemEuropean : There "
          "is minimum one doubled algorithmic dimension!");
    }
  }

  // Build the coefficient collections for the operations
  // Some of the coefficient collections are vectors, some are matrices and one is a tensor (for the
  // Up/Down Four Op Dim case)
  size_t coefficientVectorSize = this->HestonAlgoDims.size();
  this->dCoeff = new sgpp::base::DataVector(coefficientVectorSize);
  this->eCoeff = new sgpp::base::DataVector(coefficientVectorSize);
  this->fCoeff = new sgpp::base::DataVector(coefficientVectorSize);
  this->gCoeff = new sgpp::base::DataVector(coefficientVectorSize);
  this->zCoeff = new sgpp::base::DataVector(coefficientVectorSize);
  this->bCoeff = new sgpp::base::DataMatrix(coefficientVectorSize, coefficientVectorSize);
  this->cCoeff = new sgpp::base::DataMatrix(coefficientVectorSize, coefficientVectorSize);
  this->hCoeff = new sgpp::base::DataMatrix(coefficientVectorSize, coefficientVectorSize);
  this->xCoeff = new sgpp::base::DataMatrix(coefficientVectorSize, coefficientVectorSize);
  this->yCoeff = new sgpp::base::DataMatrix(coefficientVectorSize, coefficientVectorSize);
  this->wCoeff = new sgpp::base::DataMatrix(coefficientVectorSize, coefficientVectorSize);
  create4dEqualDimSizeArray(coefficientVectorSize, &(this->kCoeff));

  // create the inner grid
  // Here we have to make sure that the alpha_inner doesn't come out with zeros at the other end.
  this->GridConverter->buildInnerGridWithCoefs(*this->BoundGrid, *this->alpha_complete,
                                               &this->InnerGrid, &this->alpha_inner);

  if (bLogTransform == false) {
    // Build coefficients
    buildXCoefficients();
    buildYCoefficients();
    buildWCoefficients();
    buildZCoefficients();
    buildGCoefficients();
    buildFCoefficients();
    buildDCoefficients();

    // Create operators
    this->OpXBound = sgpp::op_factory::createOperationHestonX(
        *this->BoundGrid, *this->xCoeff).release();
    this->OpXInner = sgpp::op_factory::createOperationHestonX(
        *this->InnerGrid, *this->xCoeff).release();
    this->OpYBound = sgpp::op_factory::createOperationHestonY(
        *this->BoundGrid, *this->yCoeff).release();
    this->OpYInner = sgpp::op_factory::createOperationHestonY(
        *this->InnerGrid, *this->yCoeff).release();
    this->OpWBound = sgpp::op_factory::createOperationHestonW(
        *this->BoundGrid, *this->wCoeff).release();
    this->OpWInner = sgpp::op_factory::createOperationHestonW(
        *this->InnerGrid, *this->wCoeff).release();
    this->OpZBound = sgpp::op_factory::createOperationHestonZ(
        *this->BoundGrid, *this->zCoeff).release();
    this->OpZInner = sgpp::op_factory::createOperationHestonZ(
        *this->InnerGrid, *this->zCoeff).release();
    this->OpGBound = sgpp::op_factory::createOperationHestonGLog(
        *this->BoundGrid, *this->gCoeff).release();
    this->OpGInner = sgpp::op_factory::createOperationHestonGLog(
        *this->InnerGrid, *this->gCoeff).release();
    this->OpDBound = sgpp::op_factory::createOperationHestonDLog(
        *this->BoundGrid, *this->dCoeff).release();
    this->OpDInner = sgpp::op_factory::createOperationHestonDLog(
        *this->InnerGrid, *this->dCoeff).release();
    this->OpFBound = sgpp::op_factory::createOperationHestonFLog(
        *this->BoundGrid, *this->fCoeff).release();
    this->OpFInner = sgpp::op_factory::createOperationHestonFLog(
        *this->InnerGrid, *this->fCoeff).release();
  } else {
    // Log-transformed case
    // Build coefficients
    buildBCoefficientsLogTransform();
    buildCCoefficientsLogTransform();
    buildDCoefficientsLogTransform();
    buildECoefficientsLogTransform();
    buildFCoefficientsLogTransform();
    buildGCoefficientsLogTransform();
    buildHCoefficientsLogTransform();
    buildKCoefficientsLogTransform();

    // Create operators
    this->OpBBound = sgpp::op_factory::createOperationHestonBLog(
        *this->BoundGrid, *this->bCoeff).release();
    this->OpBInner = sgpp::op_factory::createOperationHestonBLog(
        *this->InnerGrid, *this->bCoeff).release();
    this->OpCBound = sgpp::op_factory::createOperationHestonCLog(
        *this->BoundGrid, *this->cCoeff).release();
    this->OpCInner = sgpp::op_factory::createOperationHestonCLog(
        *this->InnerGrid, *this->cCoeff).release();
    this->OpDBound = sgpp::op_factory::createOperationHestonDLog(
        *this->BoundGrid, *this->dCoeff).release();
    this->OpDInner = sgpp::op_factory::createOperationHestonDLog(
        *this->InnerGrid, *this->dCoeff).release();
    this->OpEBound = sgpp::op_factory::createOperationHestonELog(
        *this->BoundGrid, *this->eCoeff).release();
    this->OpEInner = sgpp::op_factory::createOperationHestonELog(
        *this->InnerGrid, *this->eCoeff).release();
    this->OpFBound = sgpp::op_factory::createOperationHestonFLog(
        *this->BoundGrid, *this->fCoeff).release();
    this->OpFInner = sgpp::op_factory::createOperationHestonFLog(
        *this->InnerGrid, *this->fCoeff).release();
    this->OpGBound = sgpp::op_factory::createOperationHestonGLog(
        *this->BoundGrid, *this->gCoeff).release();
    this->OpGInner = sgpp::op_factory::createOperationHestonGLog(
        *this->InnerGrid, *this->gCoeff).release();
    this->OpHBound = sgpp::op_factory::createOperationHestonHLog(
        *this->BoundGrid, *this->hCoeff).release();
    this->OpHInner = sgpp::op_factory::createOperationHestonHLog(
        *this->InnerGrid, *this->hCoeff).release();
    this->OpKBound = sgpp::op_factory::createOperationHestonKLog(
        *this->BoundGrid, &(this->kCoeff)).release();
    this->OpKInner = sgpp::op_factory::createOperationHestonKLog(
        *this->InnerGrid, &(this->kCoeff)).release();
  }

  // Create operations that are needed by both the log-transformed and Cartesian case
  this->OpAInner = sgpp::op_factory::createOperationLTwoDotProduct(*this->InnerGrid).release();
  this->OpABound = sgpp::op_factory::createOperationLTwoDotProduct(*this->BoundGrid).release();

  // right hand side if System
  this->rhs = NULL;

  // set coarsen settings
  this->useCoarsen = useCoarsen;
  this->coarsenThreshold = coarsenThreshold;
  this->refineThreshold = refineThreshold;
  this->adaptSolveMode = adaptSolveMode;
  this->numCoarsenPoints = numCoarsenPoints;
  this->refineMode = refineMode;
  this->refineMaxLevel = refineMaxLevel;

  // init Number of AverageGridPoins
  this->numSumGridpointsInner = 0;
  this->numSumGridpointsComplete = 0;

  // init option type and strike
  this->dStrike = dStrike;
  this->option_type = option_type;

  // save coordinate transformations
  this->b_log_transform = bLogTransform;
}

HestonParabolicPDESolverSystemEuroAmer::~HestonParabolicPDESolverSystemEuroAmer() {
  delete this->dCoeff;
  delete this->eCoeff;
  delete this->fCoeff;
  delete this->gCoeff;
  delete this->zCoeff;
  delete this->bCoeff;
  delete this->cCoeff;
  delete this->hCoeff;
  delete this->xCoeff;
  delete this->yCoeff;
  delete this->wCoeff;
  delete4dEqualDimSizeArray(this->HestonAlgoDims.size(), &(this->kCoeff));

  if (!b_log_transform) {
    delete this->OpXBound;
    delete this->OpXInner;
    delete this->OpYBound;
    delete this->OpYInner;
    delete this->OpWBound;
    delete this->OpWInner;
    delete this->OpZBound;
    delete this->OpZInner;
    delete this->OpGBound;
    delete this->OpGInner;
    delete this->OpDBound;
    delete this->OpDInner;
    delete this->OpFBound;
    delete this->OpFInner;
  } else {
    delete this->OpBBound;
    delete this->OpBInner;
    delete this->OpCBound;
    delete this->OpCInner;
    delete this->OpDBound;
    delete this->OpDInner;
    delete this->OpEBound;
    delete this->OpEInner;
    delete this->OpFBound;
    delete this->OpFInner;
    delete this->OpGBound;
    delete this->OpGInner;
    delete this->OpHBound;
    delete this->OpHInner;
    delete this->OpKBound;
    delete this->OpKInner;
  }

  delete this->OpAInner;
  delete this->OpABound;

  delete this->BoundaryUpdate;
  delete this->GridConverter;

  if (this->InnerGrid != NULL) {
    delete this->InnerGrid;
  }

  if (this->alpha_inner != NULL) {
    delete this->alpha_inner;
  }

  if (this->rhs != NULL) {
    delete this->rhs;
  }

  delete this->alpha_complete_old;
  delete this->alpha_complete_tmp;
  delete this->oldGridStorage;
  delete this->secondGridStorage;
}

void HestonParabolicPDESolverSystemEuroAmer::applyLOperatorComplete(
    sgpp::base::DataVector& alpha, sgpp::base::DataVector& result) {
  sgpp::base::DataVector temp(alpha.getSize());

  result.setAll(0.0);

  // Apply the riskfree rate
  if (this->r != 0.0) {
    this->OpABound->mult(alpha, temp);
    result.axpy((-1.0) * this->r, temp);
  }

  if (this->b_log_transform) {
    // Log-transformed coordinates

    // Apply the B operator
    this->OpBBound->mult(alpha, temp);
    result.sub(temp);

    // Apply the C operator
    this->OpCBound->mult(alpha, temp);
    result.sub(temp);

    // Apply the D operator
    this->OpDBound->mult(alpha, temp);
    result.sub(temp);

    // Apply the E operator
    this->OpEBound->mult(alpha, temp);
    result.add(temp);

    // Apply the F operator
    this->OpFBound->mult(alpha, temp);
    result.add(temp);

    // Apply the G operator
    this->OpGBound->mult(alpha, temp);
    result.sub(temp);

    // Apply the H operator
    this->OpHBound->mult(alpha, temp);
    result.sub(temp);

    // Apply the K operator
    this->OpKBound->mult(alpha, temp);
    result.sub(temp);
  } else {
    // Cartesian coordinates
    // Only a single asset is supported.

    // Apply the X operator
    this->OpXBound->mult(alpha, temp);
    result.sub(temp);

    // Apply the Y operator
    this->OpYBound->mult(alpha, temp);
    result.sub(temp);

    // Apply the G operator
    this->OpGBound->mult(alpha, temp);
    result.sub(temp);

    // Apply the D operator
    this->OpDBound->mult(alpha, temp);
    result.sub(temp);

    // Apply the F operator
    this->OpFBound->mult(alpha, temp);
    result.add(temp);

    // Apply the W operator
    this->OpWBound->mult(alpha, temp);
    result.sub(temp);

    // Apply the Z operator
    this->OpZBound->mult(alpha, temp);
    result.add(temp);
  }
}

void HestonParabolicPDESolverSystemEuroAmer::applyLOperatorInner(sgpp::base::DataVector& alpha,
                                                                 sgpp::base::DataVector& result) {
  sgpp::base::DataVector temp(alpha.getSize());

  result.setAll(0.0);

  // Apply the riskfree rate
  if (this->r != 0.0) {
    this->OpAInner->mult(alpha, temp);
    result.axpy((-1.0) * this->r, temp);
  }

  if (this->b_log_transform) {
    // Log-transformed coordinates

    // Apply the B method
    this->OpBInner->mult(alpha, temp);
    result.sub(temp);

    // Apply the C method
    this->OpCInner->mult(alpha, temp);
    result.sub(temp);

    // Apply the D method
    this->OpDInner->mult(alpha, temp);
    result.sub(temp);

    // Apply the E method
    this->OpEInner->mult(alpha, temp);
    result.add(temp);

    // Apply the F method
    this->OpFInner->mult(alpha, temp);
    result.add(temp);

    // Apply the G method
    this->OpGInner->mult(alpha, temp);
    result.sub(temp);

    // Apply the H method
    this->OpHInner->mult(alpha, temp);
    result.sub(temp);

    // Apply the K method
    this->OpKInner->mult(alpha, temp);
    result.sub(temp);
  } else {
    // Cartesian coordinates

    // Apply the X method
    this->OpXInner->mult(alpha, temp);
    result.sub(temp);

    // Apply the Y method
    this->OpYInner->mult(alpha, temp);
    result.sub(temp);

    // Apply the G method
    this->OpGInner->mult(alpha, temp);
    result.sub(temp);

    // Apply the D method
    this->OpDInner->mult(alpha, temp);
    result.sub(temp);

    // Apply the F method
    this->OpFInner->mult(alpha, temp);
    result.add(temp);

    // Apply the W method
    this->OpWInner->mult(alpha, temp);
    result.sub(temp);

    // Apply the Z method
    this->OpZInner->mult(alpha, temp);
    result.add(temp);
  }
}

void HestonParabolicPDESolverSystemEuroAmer::applyMassMatrixComplete(
    sgpp::base::DataVector& alpha, sgpp::base::DataVector& result) {
  sgpp::base::DataVector temp(alpha.getSize());

  result.setAll(0.0);

  // Apply the mass matrix
  this->OpABound->mult(alpha, temp);

  result.add(temp);
}

void HestonParabolicPDESolverSystemEuroAmer::applyMassMatrixInner(sgpp::base::DataVector& alpha,
                                                                  sgpp::base::DataVector& result) {
  sgpp::base::DataVector temp(alpha.getSize());

  result.setAll(0.0);

  // Apply the mass matrix
  this->OpAInner->mult(alpha, temp);

  result.add(temp);
}

void HestonParabolicPDESolverSystemEuroAmer::finishTimestep() {
  this->nExecTimesteps++;

  // Replace the inner coefficients on the boundary grid
  this->GridConverter->updateBoundaryCoefs(*this->alpha_complete, *this->alpha_inner);

#ifndef NOBOUNDARYDISCOUNT

  // Adjust the boundaries with the riskfree rate
  if (this->r != 0.0) {
    if (this->tOperationMode == "ExEul" || this->tOperationMode == "AdBas") {
      this->BoundaryUpdate->multiplyBoundary(*this->alpha_complete,
                                             exp(((-1.0) * (this->r * this->TimestepSize))));
    }
  }

#endif
}

void HestonParabolicPDESolverSystemEuroAmer::coarsenAndRefine(bool isLastTimestep) {
  // add number of Gridpoints
  this->numSumGridpointsInner += this->InnerGrid->getSize();
  this->numSumGridpointsComplete += this->BoundGrid->getSize();

  if (this->useCoarsen == true && isLastTimestep == false) {
    ///////////////////////////////////////////////////
    // Start integrated refinement & coarsening
    ///////////////////////////////////////////////////

    size_t originalGridSize = this->BoundGrid->getSize();

    // Coarsen the grid
    sgpp::base::GridGenerator& myGenerator = this->BoundGrid->getGenerator();

    if (this->adaptSolveMode == "refine" || this->adaptSolveMode == "coarsenNrefine") {
      size_t numRefines = myGenerator.getNumberOfRefinablePoints();
      sgpp::base::SurplusRefinementFunctor myRefineFunc(
          *this->alpha_complete, numRefines, this->refineThreshold);

      if (this->refineMode == "maxLevel") {
        myGenerator.refineMaxLevel(myRefineFunc, this->refineMaxLevel);
        this->alpha_complete->resizeZero(this->BoundGrid->getSize());
      }

      if (this->refineMode == "classic") {
        myGenerator.refine(myRefineFunc);
        this->alpha_complete->resizeZero(this->BoundGrid->getSize());
      }
    }

    if (this->adaptSolveMode == "coarsen" || this->adaptSolveMode == "coarsenNrefine") {
      size_t numCoarsen = myGenerator.getNumberOfRemovablePoints();
      sgpp::base::SurplusCoarseningFunctor myCoarsenFunctor(
          *this->alpha_complete, numCoarsen, this->coarsenThreshold);
      myGenerator.coarsenNFirstOnly(myCoarsenFunctor, *this->alpha_complete, originalGridSize);
    }

    ///////////////////////////////////////////////////
    // End integrated refinement & coarsening
    ///////////////////////////////////////////////////

    // rebuild the inner grid + coefficients
    this->GridConverter->rebuildInnerGridWithCoefs(*this->BoundGrid, *this->alpha_complete,
                                                   &this->InnerGrid, &this->alpha_inner);
  }
}

void HestonParabolicPDESolverSystemEuroAmer::startTimestep() {
#ifndef NOBOUNDARYDISCOUNT

  // Adjust the boundaries with the riskfree rate
  if (this->r != 0.0) {
    if (this->tOperationMode == "CrNic" || this->tOperationMode == "ImEul") {
      sgpp::base::GridStorage* storage = &this->BoundGrid->getStorage();

      for (size_t i = 0; i < storage->getSize(); i++) {
        sgpp::base::GridIndex& curPoint = (*storage)[i];

        sgpp::base::DataVector pointCoords(storage->getDimension());
        curPoint.getCoords(pointCoords);

        // Discount the boundary points
        if (!curPoint.isInnerPoint()) {
          this->alpha_complete->set(
              i, this->alpha_complete->get(i) * exp(((-1.0) * (this->r * this->TimestepSize))));
        }
      }
    }
  }

#endif
}

void HestonParabolicPDESolverSystemEuroAmer::buildGCoefficients() {
  double volvol = volvols->get(0);
  double rho = this->hMatrix->get(0, 1);
  double kappa = this->kappas->get(0);

  this->gCoeff->setAll(0.0);
  this->gCoeff->set(1, (kappa) + rho * volvol);
}

void HestonParabolicPDESolverSystemEuroAmer::buildFCoefficients() {
  double volvol = volvols->get(0);
  double kappa = this->kappas->get(0);
  double theta = this->thetas->get(0);

  this->fCoeff->setAll(0.0);
  this->fCoeff->set(1, kappa * theta - 0.5 * pow(volvol, 2.0));
}

void HestonParabolicPDESolverSystemEuroAmer::buildDCoefficients() {
  double volvol = volvols->get(0);

  this->dCoeff->setAll(0.0);
  this->dCoeff->set(1, (0.5) * pow(volvol, 2.0));
}

void HestonParabolicPDESolverSystemEuroAmer::buildXCoefficients() {
  this->xCoeff->setAll(0.0);
  this->xCoeff->set(0, 1, 1.0);
}

void HestonParabolicPDESolverSystemEuroAmer::buildWCoefficients() {
  double volvol = volvols->get(0);
  double rho = this->hMatrix->get(0, 1);

  this->wCoeff->setAll(0.0);
  this->wCoeff->set(0, 1, rho * volvol);
}

void HestonParabolicPDESolverSystemEuroAmer::buildZCoefficients() {
  this->zCoeff->setAll(0.0);
  this->zCoeff->set(0, this->r);
}

void HestonParabolicPDESolverSystemEuroAmer::buildYCoefficients() {
  this->yCoeff->setAll(0.0);
  this->yCoeff->set(0, 1, 0.5);
}

void HestonParabolicPDESolverSystemEuroAmer::buildBCoefficientsLogTransform() {
  this->bCoeff->setAll(0.0);

  for (size_t i = 0; i < this->nAssets; i++) {
    this->bCoeff->set(2 * i, 2 * i + 1, 0.5);  // Coefficient is a constant of 0.5
  }
}

void HestonParabolicPDESolverSystemEuroAmer::buildCCoefficientsLogTransform() {
  double volvol, rho;
  this->cCoeff->setAll(0.0);

  for (size_t i = 0; i < this->nAssets; i++) {
    // Determine and set the coefficient for this asset
    volvol = volvols->get(i);
    rho = this->hMatrix->get(2 * i, 2 * i + 1);
    this->cCoeff->set(2 * i, 2 * i + 1, volvol * rho);
  }
}

void HestonParabolicPDESolverSystemEuroAmer::buildDCoefficientsLogTransform() {
  this->dCoeff->setAll(0.0);

  for (size_t i = 0; i < this->nAssets; i++) {
    // Determine and set the coefficient for this asset
    double volvol = volvols->get(i);
    this->dCoeff->set(2 * i + 1, (0.5) * pow(volvol, 2.0));
  }
}

void HestonParabolicPDESolverSystemEuroAmer::buildECoefficientsLogTransform() {
  this->eCoeff->setAll(0.0);

  for (size_t i = 0; i < this->nAssets; i++) {
    this->eCoeff->set(2 * i, this->r);  // Coefficient is independent of which asset it is.
  }
}

void HestonParabolicPDESolverSystemEuroAmer::buildFCoefficientsLogTransform() {
  double theta, kappa, volvol;
  this->fCoeff->setAll(0.0);

  for (size_t i = 0; i < nAssets; i++) {
    theta = this->thetas->get(i);
    kappa = this->kappas->get(i);
    volvol = this->volvols->get(i);
    this->fCoeff->set(2 * i + 1, kappa * theta - 0.5 * pow(volvol, 2.0));
  }
}

void HestonParabolicPDESolverSystemEuroAmer::buildGCoefficientsLogTransform() {
  this->gCoeff->setAll(0.0);
  double kappa;

  for (size_t i = 0; i < nAssets; i++) {
    kappa = this->kappas->get(i);
    this->gCoeff->set(2 * i + 1, kappa);
  }
}

void HestonParabolicPDESolverSystemEuroAmer::buildHCoefficientsLogTransform() {
  this->hCoeff->setAll(0.0);

  for (size_t i = 0; i < nAssets; i++) {
    this->hCoeff->set(2 * i, 2 * i + 1, 0.5);
  }
}

void HestonParabolicPDESolverSystemEuroAmer::buildKCoefficientsLogTransform() {
  double rho;
  setAll4dEqualDimSizeArray(this->HestonAlgoDims.size(), &(this->kCoeff), 0.0);

  for (size_t i = 0; i < nAssets; i++) {
    for (size_t j = i + 1; j < nAssets; j++) {
      rho = this->hMatrix->get(2 * i, 2 * j);
      this->kCoeff[2 * i][2 * i + 1][2 * j][2 * j + 1] = rho;
    }
  }
}

void HestonParabolicPDESolverSystemEuroAmer::create4dEqualDimSizeArray(size_t dimSize,
                                                                       double***** array) {
  (*array) = reinterpret_cast<double****>(calloc(dimSize, sizeof(double***)));

  for (size_t i = 0; i < dimSize; i++) {
    (*array)[i] = reinterpret_cast<double***>(calloc(dimSize, sizeof(double**)));

    for (size_t j = 0; j < dimSize; j++) {
      (*array)[i][j] = reinterpret_cast<double**>(calloc(dimSize, sizeof(double*)));

      for (size_t k = 0; k < dimSize; k++) {
        (*array)[i][j][k] = reinterpret_cast<double*>(calloc(dimSize, sizeof(double)));

        for (size_t m = 0; m < dimSize; m++) {
          (*array)[i][j][k][m] = 0.0;
        }
      }
    }
  }
}

void HestonParabolicPDESolverSystemEuroAmer::delete4dEqualDimSizeArray(size_t dimSize,
                                                                       double***** array) {
  for (size_t i = 0; i < dimSize; i++) {
    for (size_t j = 0; j < dimSize; j++) {
      for (size_t k = 0; k < dimSize; k++) {
        delete[](*array)[i][j][k];
      }

      delete[](*array)[i][j];
    }

    delete[](*array)[i];
  }

  delete[](*array);
}

void HestonParabolicPDESolverSystemEuroAmer::setAll4dEqualDimSizeArray(size_t dimSize,
                                                                       double***** array,
                                                                       double value) {
  for (size_t i = 0; i < dimSize; i++) {
    for (size_t j = 0; j < dimSize; j++) {
      for (size_t k = 0; k < dimSize; k++) {
        for (size_t m = 0; m < dimSize; m++) {
          (*array)[i][j][k][m] = value;
        }
      }
    }
  }
}
}  // namespace finance
}  // namespace sgpp
