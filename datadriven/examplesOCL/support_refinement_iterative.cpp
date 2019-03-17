#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <vector>

#include "sgpp/base/grid/Grid.hpp"
#include "sgpp/base/grid/GridStorage.hpp"
#include "sgpp/base/grid/generation/GridGenerator.hpp"
#include "sgpp/base/grid/storage/hashmap/HashGridPoint.hpp"
#include "sgpp/datadriven/tools/ARFFTools.hpp"

#include "sgpp/datadriven/grid/support_refinement_iterative.hpp"

int main(void) {
  std::string dataset_file_name("datasets_diss/gaussian_c4_size500_dim2id2_noise.arff");

  sgpp::datadriven::Dataset dataset = sgpp::datadriven::ARFFTools::readARFF(dataset_file_name);
  int64_t entries = dataset.getNumberInstances();
  int64_t dim = dataset.getDimension();
  std::vector<double> data = std::move(dataset.getData());  // this won't actually move

  std::cout << "dim: " << dim << " entries: " << entries << std::endl;

  int64_t max_level = 10;
  int64_t min_support = 40;
  std::string grid_file_name("results_diss/support_refined_grid_G2D.csv");

  sgpp::datadriven::support_refinement_iterative ref(dim, max_level, min_support, data);
  ref.enable_OCL("OCL_configs/config_ocl_float_QuadroGP100.cfg");
  ref.refine();

  std::vector<int64_t> &ls = ref.get_levels();
  std::vector<int64_t> &is = ref.get_indices();

  std::unique_ptr<sgpp::base::Grid> grid =
      std::unique_ptr<sgpp::base::Grid>(sgpp::base::Grid::createLinearGrid(dim));
  // sgpp::base::GridGenerator &grid_generator = grid->getGenerator();
  sgpp::base::GridStorage &grid_storage = grid->getStorage();
  sgpp::base::HashGridPoint p(dim);
  for (int64_t gp = 0; gp < static_cast<int64_t>(ls.size() / dim); gp += 1) {
    for (int64_t d = 0; d < dim; d += 1) {
      p.set(d, ls[gp * dim + d], is[gp * dim + d]);
    }
    grid_storage.insert(p);
  }

  if (ls.size() > 0) {
    std::cout << "final number of grid points created: " << (ls.size() / dim) << std::endl;
    sgpp::datadriven::support_refinement_iterative::write_grid_positions(dim, grid_file_name,
                                                                         dataset_file_name, ls, is);
  } else {
    std::cerr << "error: did not create any grid points" << std::endl;
  }
  return 1;
}