// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#ifndef XDPHIPHIDOWNBBLINEAR_HPP
#define XDPHIPHIDOWNBBLINEAR_HPP

#include <sgpp/base/grid/GridStorage.hpp>
#include <sgpp/base/datatypes/DataVector.hpp>

#include <sgpp/globaldef.hpp>

namespace sgpp {
namespace finance {

/**
 * Implementation of sweep operator (): 1D Down for
 * Bilinearform \f$\int_{x} x \frac{\partial \phi(x)}{x} \phi(x) dx\f$
 */
class XdPhiPhiDownBBLinear {
 protected:
  typedef sgpp::base::GridStorage::grid_iterator grid_iterator;

  /// Pointer to the sgpp::base::GridStorage Object
  sgpp::base::GridStorage* storage;
  /// Pointer to the bounding box Obejct
  sgpp::base::BoundingBox* boundingBox;

 public:
  /**
   * Constructor
   *
   * @param storage the grid's sgpp::base::GridStorage object
   */
  explicit XdPhiPhiDownBBLinear(sgpp::base::GridStorage* storage);

  /**
   * Destructor
   */
  ~XdPhiPhiDownBBLinear();

  /**
   * This operations performs the calculation of down in the direction of dimension <i>dim</i>
   * on a grid with fix Dirichlet 0 boundary conditions
   *
   * @param source sgpp::base::DataVector that contains the gridpoint's coefficients (values from
   * the vector of the laplace operation)
   * @param result sgpp::base::DataVector that contains the result of the down operation
   * @param index a iterator object of the grid
   * @param dim current fixed dimension of the 'execution direction'
   */
  void operator()(sgpp::base::DataVector& source, sgpp::base::DataVector& result,
                  grid_iterator& index, size_t dim);

 protected:
  /**
   * recursive function for the calculation of Down without Bounding Box support
   *
   * @param source sgpp::base::DataVector that contains the coefficients of the ansatzfunction
   * @param result sgpp::base::DataVector in which the result of the operation is stored
   * @param index reference to a griditerator object that is used navigate through the grid
   * @param dim the dimension in which the operation is executed
   * @param fl function value on the left boundary
   * @param fr function value on the right boundary
   */
  void rec(sgpp::base::DataVector& source, sgpp::base::DataVector& result, grid_iterator& index,
           size_t dim, double fl, double fr);

  /**
   * recursive function for the calculation of Down wit Bounding Box support
   *
   * @param source sgpp::base::DataVector that contains the coefficients of the ansatzfunction
   * @param result sgpp::base::DataVector in which the result of the operation is stored
   * @param index reference to a griditerator object that is used navigate through the grid
   * @param dim the dimension in which the operation is executed
   * @param fl function value on the left boundary
   * @param fr function value on the right boundary
   * @param q interval width in the current dimension <i>dim</i>
   * @param t interval offset in current dimension <i>dim</i>
   */
  void recBB(sgpp::base::DataVector& source, sgpp::base::DataVector& result, grid_iterator& index,
             size_t dim, double fl, double fr, double q, double t);
};

}  // namespace finance
}  // namespace sgpp

#endif /* XDPHIPHIDOWNBBLINEAR_HPP */
