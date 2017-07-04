/* Copyright (C) 2008-today The SG++ project
 * This file is part of the SG++ project. For conditions of distribution and
 * use, please see the copyright notice provided with SG++ or at
 * sgpp.sparsegrids.org
 *
 * datadriven_refinement.cpp
 *
 *  Created on: 20.04.2017
 *      Author: David Pfander
 */

#include <cstddef>
#include <fstream>
#include <string>

#include "sgpp/base/grid/Grid.hpp"
#include "sgpp/base/grid/GridStorage.hpp"
#include "sgpp/datadriven/DatadrivenOpFactory.hpp"
#include "sgpp/datadriven/operation/hash/OperationDatadrivenRefinement/RefinementFunctionalFactory.hpp"
#include "sgpp/datadriven/tools/ARFFTools.hpp"
#include "sgpp/datadriven/tools/Dataset.hpp"

void write_csv(std::ostream& stream, sgpp::base::GridStorage& storage) {
  size_t dim = storage.getDimension();
  for (size_t d = 0; d < dim; d++) {
    if (d > 0) {
      stream << ", ";
    }
    stream << "level_" << d;
  }
  for (size_t d = 0; d < dim; d++) {
    stream << ", ";
    stream << "index_" << d;
  }
  stream << std::endl;

  for (size_t i = 0; i < storage.getSize(); i++) {
    sgpp::base::GridPoint& p = storage[i];
    for (size_t d = 0; d < dim; d++) {
      if (d > 0) {
        stream << ", ";
      }
      stream << p.getLevel(d);
    }
    for (size_t d = 0; d < dim; d++) {
      stream << ", ";
      stream << p.getIndex(d);
    }
    stream << std::endl;
  }
}

int main(int argc, char** argv) {
  const std::string filename("dataset2_dim2.arff");

  // read dataset
  sgpp::datadriven::Dataset dataset = sgpp::datadriven::ARFFTools::readARFF(filename);
  sgpp::base::DataMatrix& training_data = dataset.getData();
  sgpp::base::DataVector& training_data_values = dataset.getTargets();

  const size_t dim = dataset.getDimension();
  const size_t start_level = 1;

  // create grid
  std::shared_ptr<sgpp::base::Grid> grid =
      std::shared_ptr<sgpp::base::Grid>(sgpp::base::Grid::createLinearGrid(dim));
  sgpp::base::GridGenerator& generator = grid->getGenerator();
  generator.regular(start_level);

  size_t refine_max_level = 8;
  size_t refine_max_level_sum = 13;
  size_t support_points_for_accept = 50;

  std::function<bool(sgpp::base::DataMatrix&, sgpp::base::GridPoint&)> f =
      sgpp::datadriven::datadriven_refinement_factory::createMaxLevelAndLevelSumFunctor(
          refine_max_level, refine_max_level_sum, support_points_for_accept);

  std::shared_ptr<sgpp::datadriven::OperationDatadrivenRefinement> refine_op =
      std::shared_ptr<sgpp::datadriven::OperationDatadrivenRefinement>(
          sgpp::op_factory::createOperationDatadrivenRefinement(f));

  refine_op->refine(*grid, training_data, training_data_values);

  auto& storage = grid->getStorage();
  for (size_t i = 0; i < storage.getSize(); i++) {
    sgpp::base::GridPoint& p = storage[i];
    // std::cout << p << std::endl;
    p.printLevelIndex(std::cout) << std::endl;
  }

  std::cout << "total grid points: " << storage.getSize() << std::endl;

  std::ofstream myfile;
  myfile.open("grid.csv");
  write_csv(myfile, storage);
  myfile.close();

  return 0;
}
