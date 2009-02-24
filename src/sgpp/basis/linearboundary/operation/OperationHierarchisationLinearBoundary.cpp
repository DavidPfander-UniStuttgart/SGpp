 /*****************************************************************************/
/* This file is part of sgpp, a program package making use of spatially      */
/* adaptive sparse grids to solve numerical problems                         */
/*                                                                           */
/* Copyright (C) 2009 Alexander Heinecke (Alexander.Heinecke@mytum.de)       */
/*                                                                           */
/* sgpp is free software; you can redistribute it and/or modify              */
/* it under the terms of the GNU General Public License as published by      */
/* the Free Software Foundation; either version 3 of the License, or         */
/* (at your option) any later version.                                       */
/*                                                                           */
/* sgpp is distributed in the hope that it will be useful,                   */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of            */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             */
/* GNU General Public License for more details.                              */
/*                                                                           */
/* You should have received a copy of the GNU General Public License         */
/* along with sgpp; if not, write to the Free Software                       */
/* Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA */
/* or see <http://www.gnu.org/licenses/>.                                    */
/*****************************************************************************/

#include "basis/linearboundary/operation/OperationHierarchisationLinearBoundary.hpp"
#include "basis/linearboundary/algorithm_sweep/HierarchisationLinearBoundary.hpp"
#include "basis/linearboundary/algorithm_sweep/DehierarchisationLinearBoundary.hpp"

#include "sgpp.hpp"

#include "basis/basis.hpp"
#include "data/DataVector.h"

namespace sg
{

/**
 * Implements the hierarchisation on a sprase grid with linear base functions and boundaries
 *
 * @param node_values the functions values in the node base
 */
void OperationHierarchisationLinearBoundary::doHierarchisation(DataVector& node_values)
{
	detail::HierarchisationLinearBoundary func(this->storage);
	sweep<detail::HierarchisationLinearBoundary> s(func, this->storage);

	if (this->storage->dim() > 1)
	{
		// loop over the zero dim
		for (size_t zd = 0; zd < this->storage->dim(); zd++)
		{
			// do hierarchisation on the remaining grid
			for (size_t i = 0; i < this->storage->dim(); i++)
			{
				if (i != zd)
				{
					// hierarchisation of the grid's left boundary in direction of dimension dim-(i+1)
					s.sweep1D_Boundary(node_values, node_values, i, zd, true);
					// hierarchisation of the grid's right boundary in direction of dimension dim-(i+1)
					s.sweep1D_Boundary(node_values, node_values, i, zd, false);
					// hierarchisation of the grid's core in direction of dimension dim-(i+1)
					s.sweep1D(node_values, node_values, i);
				}
			}
		}
	}
	else
	{
		// hierarchisation of the grid's core in direction of dimension dim-(i+1)
		s.sweep1D(node_values, node_values, 0);
	}
}

/**
 * Implements the dehierarchisation on a sprase grid with linear base functions and boundaries
 *
 * @param alpha the coefficients of the sparse grid's base functions
 */
void OperationHierarchisationLinearBoundary::doDehierarchisation(DataVector& alpha)
{
	detail::DehierarchisationLinearBoundary func(this->storage);
	sweep<detail::DehierarchisationLinearBoundary> s(func, this->storage);

	if (this->storage->dim() > 1)
	{
		// loop over the zero dim
		for (size_t zd = 0; zd < this->storage->dim(); zd++)
		{
			// do dehierarchisation on the remaining grid
			for (size_t i = 0; i < this->storage->dim(); i++)
			{
				if (i != zd)
				{
					// dehierarchisation of the grid's core in direction of dimension dim-(i+1)
					s.sweep1D(alpha, alpha, (this->storage->dim()-(i+1)));
					// dehierarchisation of the grid's right boundary in direction of dimension dim-(i+1)
					s.sweep1D_Boundary(alpha, alpha, (this->storage->dim()-(i+1)), (this->storage->dim()-(zd+1)), false);
					// dehierarchisation of the grid's left boundary in direction of dimension dim-(i+1)
					s.sweep1D_Boundary(alpha, alpha, (this->storage->dim()-(i+1)), (this->storage->dim()-(zd+1)), true);
				}
			}
		}
	}
	else
	{
		// dehierarchisation of the grid's core in direction of dimension dim-(i+1)
		s.sweep1D(alpha, alpha, 0);
	}
}

}
