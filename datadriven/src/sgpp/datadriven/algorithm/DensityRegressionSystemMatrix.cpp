// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#include <sgpp/datadriven/algorithm/DensityRegressionSystemMatrix.hpp>
#include <sgpp/base/operation/BaseOpFactory.hpp>
#include <sgpp/pde/operation/PdeOpFactory.hpp>
#include <sgpp/base/exception/operation_exception.hpp>
#include <sgpp/pde/operation/hash/OperationLTwoDotProductLinear.hpp>
#include <sgpp/base/grid/GridStorage.hpp>

#include <sgpp/base/datatypes/DataVector.hpp>
#include <sgpp/base/datatypes/DataMatrix.hpp>

#include <sgpp/globaldef.hpp>

namespace SGPP {
namespace datadriven {

DensityRegressionSystemMatrix::DensityRegressionSystemMatrix(SGPP::datadriven::HistogramTree::Node &piecewiseRegressor,
SGPP::base::Grid& grid, SGPP::base::OperationMatrix& C, float_t lambdaRegression) :
        piecewiseRegressor(piecewiseRegressor), grid(grid) {
    this->lambda = lambdaRegression;

    this->A = SGPP::op_factory::createOperationLTwoDotProduct(grid);
//      this->B = SGPP::op_factory::createOperationMultipleEval(grid, *(this->data));
    this->C = &C;
}

void DensityRegressionSystemMatrix::mult(SGPP::base::DataVector& alpha, SGPP::base::DataVector& result) {
    result.setAll(0.0);

    // A * alpha
    this->A->mult(alpha, result);

    // C * alpha
    base::DataVector tmp(result.getSize());
    this->C->mult(alpha, tmp);

    // A * alpha + lambda * C * alpha
    result.axpy(this->lambda, tmp);
}

// Matrix-Multiplikation verwenden
void DensityRegressionSystemMatrix::generateb(SGPP::base::DataVector& rhs) {
//      SGPP::base::DataVector y(this->data->getNrows());
//      y.setAll(1.0);
//      // Bt * 1
//      this->B->multTranspose(y, rhs);
//      // 1 / 2M * Bt * 1
//      rhs.mult(1. / (float_t)this->data->getNrows());

//store result in rhs!
    SGPP::base::GridStorage *storage = grid.getStorage();
    for (size_t gridIndex = 0; gridIndex < storage->size(); gridIndex++) {
        SGPP::base::GridIndex *gridPoint = storage->get(gridIndex);
        rhs[gridIndex] = piecewiseRegressor.integrate(*gridPoint);
//        std::cout << "rhs[" << gridIndex << "] = " << rhs[gridIndex] << std::endl;
    }
}

DensityRegressionSystemMatrix::~DensityRegressionSystemMatrix() {
    delete this->A;
//      delete this->B;
}

}
}
