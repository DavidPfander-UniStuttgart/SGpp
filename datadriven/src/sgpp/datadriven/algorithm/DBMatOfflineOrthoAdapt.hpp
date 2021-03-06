// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#pragma once

#include <sgpp/base/exception/algorithm_exception.hpp>
#include <sgpp/datadriven/algorithm/DBMatOffline.hpp>

#include <string>

namespace sgpp {
namespace datadriven {

class DBMatOfflineOrthoAdapt : public DBMatOffline {
 public:
  /**
   * Constructor
   * Builds DBMatOfflineOrthoAdapt Object from configuration
   *
   * @param gridConfig The configuration of the grid
   * @param adaptivityConfig The configuration of the grid adaptivity
   * @param regularizationConfig The configuration of the grid regularization
   * @param densityEstimationConfig The configuration of the matrix decomposition
   */
  explicit DBMatOfflineOrthoAdapt(
      const sgpp::base::RegularGridConfiguration& gridConfig,
      const sgpp::base::AdpativityConfiguration& adaptivityConfig,
      const sgpp::datadriven::RegularizationConfiguration& regularizationConfig,
      const sgpp::datadriven::DensityEstimationConfiguration& densityEstimationConfig);

  /**
   * Constructor
   * Builds object from serialized offline object, stored in file
   *
   * @param fileName path to the file storing object
   */
  explicit DBMatOfflineOrthoAdapt(const std::string& fileName);

  DBMatOffline* clone();

  bool isRefineable() override;

  /**
   * Builds the left hand side matrix without the regularization term
   */
  void buildMatrix();

  /**
   * Decomposes and inverts the lhsMatrix of the offline object
   * (lhs + lambda*I)^{-1} = Q * (T + lambda*I)^{-1} * Q^t = Q * T_inv * Q^t
   *
   * The matrix lhsMatrix of the parent object will be altered during the process
   */
  void decomposeMatrix();

  /**
   * Decomposes the lhsMatrix into lhs = Q * T * Q^t and stores the orthogonal
   * matrix Q into the member q_ortho_matrix_. The information to reconstruct T
   * is written into diag and subdiag
   *
   * @param diag Diagonal entries of T
   * @param subdiag Sub- and superdiagonal entries of T (symmetric)
   */
  void hessenberg_decomposition(sgpp::base::DataVector& diag, sgpp::base::DataVector& subdiag);

  /**
   * Inverts a symmetric tridiagonal matrix T, which is given in the form of
   * its diagonal and subdiagonal vectors. When finished, diag and subdiag no
   * longer hold their initial values.
   *
   * @param diag Diagonal entries of T
   * @param subdiag Sub- and superdiagonal entries of T (symmetric)
   */
  void invert_symmetric_tridiag(sgpp::base::DataVector& diag, sgpp::base::DataVector& subdiag);

  /**
   * Serializes the DBMatOfflineOrthoAdapt object
   *
   * q_ortho_matrix_ and t_inv_tridiag_ are stored into the specified file,
   * which is the explicit representation of the decomposition needed for the
   * online phase
   *
   * @param fileName path where to store the file
   */
  void store(const std::string& fileName);

  // getter and setter
  size_t& getDimA() { return this->dim_a; }

  double getLambda() { return this->lambda; }

  sgpp::base::DataMatrix& getQ() { return this->q_ortho_matrix_; }

  sgpp::base::DataMatrix& getTinv() { return this->t_tridiag_inv_matrix_; }

 protected:
  size_t dim_a;                                  // quadratic matrix size of matrix to decompose
  double lambda;                                 // regularization parameter
  sgpp::base::DataMatrix q_ortho_matrix_;        // orthogonal matrix of decomposition
  sgpp::base::DataMatrix t_tridiag_inv_matrix_;  // inverse of the tridiag matrix of decomposition
};
}  // namespace datadriven
}  // namespace sgpp
