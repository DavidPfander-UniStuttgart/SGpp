
#include <boost/program_options.hpp>

#include "sgpp/base/grid/Grid.hpp"
#include "sgpp/base/grid/GridStorage.hpp"
#include "sgpp/base/grid/generation/GridGenerator.hpp"

using namespace sgpp;

int main(int argc, char **argv) {

  boost::program_options::options_description description("Allowed options");

  size_t level;
  size_t dim;

  std::string file_prefix;

  description.add_options()("help", "display help")(
      "level", boost::program_options::value<size_t>(&level)->default_value(4),
      "level of the sparse grid")(
      "dim", boost::program_options::value<size_t>(&dim)->default_value(10),
      "dimension of the sparse grid")(
      "file_prefix", boost::program_options::value<std::string>(&file_prefix),
      "write the levels and indices");

  boost::program_options::variables_map variables_map;
  boost::program_options::parsed_options options =
      parse_command_line(argc, argv, description);
  boost::program_options::store(options, variables_map);
  boost::program_options::notify(variables_map);

  if (variables_map.count("help")) {
    std::cout << description << std::endl;
    return 0;
  }

  if (variables_map.count("file_prefix") == 0) {
    std::cerr << "error: no file_prefix specified" << std::endl;
    return 1;
  }

  std::unique_ptr<base::Grid> grid;

  grid = std::unique_ptr<base::Grid>(base::Grid::createLinearGrid(dim));
  base::GridGenerator &grid_generator = grid->getGenerator();
  grid_generator.regular(level);
  std::cout << "grid created, grid points: " << grid->getSize() << std::endl;
  std::ofstream out_grid(file_prefix + "_density_grid.serialized");
  grid->serialize(out_grid);
  return 0;
}
