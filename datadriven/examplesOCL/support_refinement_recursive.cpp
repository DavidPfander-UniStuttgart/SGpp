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

#include "sgpp/datadriven/grid/support_refinement_recursive.hpp"

// // read into vector of size rows*cols
// class csv_reader {
//  private:
//   static const char delim = ',';
//   int64_t dim;
//   int64_t entries;
//   bool has_targets;
//   std::vector<double> data;
//   std::vector<double> targets;

//  public:
//   csv_reader(const std::string &file_name) : dim(0), entries(0), has_targets(false) {
//     std::ifstream s(file_name);
//     std::string line;
//     std::string value;
//     const std::string attr_prefix("@ATTRIBUTE x");
//     const std::string class_attr_prefix("@ATTRIBUTE class");
//     while (std::getline(s, line)) {
//       if (line.compare(0, attr_prefix.size(), attr_prefix) == 0) {
//         dim += 1;
//       }
//       if (line.compare(0, class_attr_prefix.size(), class_attr_prefix) == 0) {
//         has_targets = true;
//       }
//       if (line.compare("@DATA") == 0) {
//         break;
//       }
//     }
//     // std::cout << "dim: " << dim << std::endl;
//     while (std::getline(s, line)) {
//       entries += 1;
//       // std::cout << "line: " << line << std::endl;
//       std::stringstream ss;
//       ss << line;
//       int64_t cur_dim = 0;
//       while (std::getline(ss, value, delim)) {
//         // std::cout << "val: " << value << std::endl;
//         if (cur_dim < dim) {
//           data.push_back(std::stod(value));
//         } else {
//           targets.push_back(std::stod(value));
//         }
//         cur_dim += 1;
//       }
//     }
//   }

//   int64_t get_dim() { return dim; }
//   int64_t get_entries() { return entries; }
//   std::vector<double> &get_data() { return data; }
//   std::vector<double> &get_targets() { return targets; }

//   void print_data() {
//     for (int64_t i = 0; i < entries; i += 1) {
//       for (int64_t d = 0; d < dim; d += 1) {
//         if (d > 0) {
//           std::cout << ", ";
//         }
//         std::cout << data[i * dim + d];
//       }
//       std::cout << std::endl;
//     }
//   }

//   void print_targets() {
//     for (int64_t i = 0; i < entries; i += 1) {
//       std::cout << targets[i] << std::endl;
//     }
//   }
// };

// class level_generator {
//  private:
//   int64_t dim;
//   int64_t n;
//   std::vector<int64_t> cur;
//   int64_t cur_d;
//   int64_t sum;

//  public:
//   level_generator(int64_t dim, int64_t n) : dim(dim), n(n), cur(dim, 1), cur_d(0), sum(dim - 1) {
//     cur[0] = 0;  // first next gives 1, ..., 1
//   }
//   bool next() {
//     while (cur_d < dim) {
//       if (cur[cur_d] < n) {
//         cur[cur_d] += 1;
//         sum += 1;
//         for (int64_t lower_d = 0; lower_d < cur_d; lower_d += 1) {
//           sum -= cur[lower_d] + 1;
//           cur[lower_d] = 1;
//         }
//         cur_d = 0;
//         if (sum <= n + dim - 1) {
//           return true;
//         }
//       } else {
//         cur_d += 1;
//       }
//     }
//     return false;
//   }
//   int64_t get(int64_t d) { return cur[d]; }

//   std::vector<int64_t> &get() { return cur; }
// };

// class index_generator {
//  private:
//   int64_t dim;
//   std::vector<int64_t> max;
//   std::vector<int64_t> cur;
//   int64_t cur_d;

//  public:
//   index_generator(int64_t dim, std::vector<int64_t> level)
//       : dim(dim), max(dim), cur(dim), cur_d(0) {
//     for (int64_t d = 0; d < dim; d += 1) {
//       max[d] = std::pow(2, level[d]) - 1;
//     }
//     for (int64_t d = 0; d < dim; d += 1) {
//       cur[d] = 1;
//     }
//     cur[0] = -1;  // first next() gives 1, ..., 1
//   }

//   bool next() {
//     while (cur_d < dim) {
//       if (cur[cur_d] < max[cur_d]) {
//         cur[cur_d] += 2;
//         for (int64_t lower_d = 0; lower_d < cur_d; lower_d += 1) {
//           cur[lower_d] = 1;
//         }
//         cur_d = 0;
//         return true;
//       } else {
//         cur_d += 1;
//       }
//     }
//     return false;
//   }

//   int64_t get(int64_t d) { return cur[d]; }

//   std::vector<int64_t> &get() { return cur; }
// };

// class support_refinement_recursive {
//  private:
//   int64_t dim;
//   int64_t max_level;
//   int64_t min_support;
//   const std::vector<double> &data;

//   int64_t entries;
//   std::vector<std::vector<int64_t>> levels;
//   std::vector<std::vector<int64_t>> indices;
//   std::set<std::pair<std::vector<int64_t>, std::vector<int64_t>>> visited;
//   int64_t num_visited;

//   enum class direction { left, right };

//   void child_d_dir(std::vector<int64_t> &l, std::vector<int64_t> &i, std::vector<int64_t>
//   &child_l,
//                    std::vector<int64_t> &child_i, int64_t d, direction dir) {
//     child_l = l;
//     child_i = i;
//     child_l[d] += 1;
//     if (dir == direction::left) {
//       child_i[d] = 2 * child_i[d] - 1;
//     } else if (dir == direction::right) {
//       child_i[d] = 2 * child_i[d] + 1;
//     } else {
//       throw;  // never
//     }
//   }

//   void neighbor_pos_dir(std::vector<int64_t> &l, std::vector<int64_t> &i,
//                         std::vector<double> &neighbor, direction dir) {
//     if (dir == direction::left) {
//       for (int64_t d = 0; d < dim; d += 1) {
//         neighbor[d] = std::pow(2.0, static_cast<double>(-l[d])) * static_cast<double>(i[d] -
//         1ll);
//       }
//     } else if (dir == direction::right) {
//       for (int64_t d = 0; d < dim; d += 1) {
//         neighbor[d] = std::pow(2.0, static_cast<double>(-l[d])) * static_cast<double>(i[d] +
//         1ll);
//       }
//     } else {
//       throw;  // never
//     }
//   }

//   bool has_support(std::vector<double> &bound_left, std::vector<double> &bound_right,
//                    std::vector<double> &data_point) {
//     for (int64_t d = 0; d < dim; d += 1) {
//       if (data_point[d] < bound_left[d] || data_point[d] > bound_right[d]) {
//         return false;
//       }
//     }
//     return true;
//   }

//   // N * m data points accessed (~memory operations)
//   void refine_impl(std::vector<int64_t> cur_l, std::vector<int64_t> cur_i,
//                    const std::vector<double> &data) {
//     // already visited?
//     num_visited += 1;
//     if (num_visited % 1000 == 0) {
//       std::cout << "num_visited: " << num_visited << std::endl;
//     }

//     auto p = std::make_pair(cur_l, cur_i);
//     if (visited.find(std::make_pair(cur_l, cur_i)) != visited.end()) {
//       return;
//     }
//     visited.insert(p);

//     // max_level reached?
//     int64_t sum_l = 0;
//     for (int64_t d = 0; d < dim; d += 1) {
//       sum_l += cur_l[d];
//     }
//     if (sum_l > max_level + dim - 1) {
//       return;
//     }
//     // validate that the current grid point has enough grid points on its
//     // support
//     std::vector<double> bound_left(dim);
//     neighbor_pos_dir(cur_l, cur_i, bound_left, direction::left);
//     std::vector<double> bound_right(dim);
//     neighbor_pos_dir(cur_l, cur_i, bound_right, direction::right);
//     std::vector<double> data_point(dim);
//     // std::cout << "---------- l, i, n_left, n_right, data points -----------"
//     //           << std::endl;
//     // print_level(cur_l);
//     // print_index(cur_i);
//     // print_pos(bound_left);
//     // print_pos(bound_right);
//     int64_t num_support = 0;
//     int64_t cur_entries = data.size() / dim;
//     for (int64_t i = 0; i < cur_entries; i += 1) {
//       for (int64_t d = 0; d < dim; d += 1) {
//         data_point[d] = data[i * dim + d];
//       }
//       if (has_support(bound_left, bound_right, data_point)) {
//         num_support += 1;
//       }
//     }
//     // std::cout << "num_support: " << num_support << std::endl;
//     if (num_support < min_support) {
//       return;
//     }

//     std::vector<double> filtered_dataset;
//     bool use_filtered = false;
//     if (num_support < 0.5 * data.size()) {
//       use_filtered = true;
//       for (int64_t i = 0; i < cur_entries; i += 1) {
//         for (int64_t d = 0; d < dim; d += 1) {
//           data_point[d] = data[i * dim + d];
//         }
//         if (has_support(bound_left, bound_right, data_point)) {
//           for (int64_t d = 0; d < dim; d += 1) {
//             filtered_dataset.push_back(data[i * dim + d]);
//           }
//         }
//       }
//     }

//     levels.push_back(cur_l);
//     indices.push_back(cur_i);

//     if (levels.size() % 100 == 0) {
//       std::cout << "reached size: " << levels.size() << std::endl;
//     }
//     // recurse children
//     // TODO: need strategy to avoid duplicate recursive calls (or simply use
//     // hashmap)
//     std::vector<int64_t> child_l(dim);
//     std::vector<int64_t> child_i(dim);
//     for (int64_t d = 0; d < dim; d += 1) {
//       child_d_dir(cur_l, cur_i, child_l, child_i, d, direction::left);
//       // if () {
//       if (use_filtered) {
//         refine_impl(child_l, child_i, filtered_dataset);
//       } else {
//         refine_impl(child_l, child_i, data);
//       }
//       // }
//       child_d_dir(cur_l, cur_i, child_l, child_i, d, direction::right);
//       // if () {
//       if (use_filtered) {
//         refine_impl(child_l, child_i, filtered_dataset);
//       } else {
//         refine_impl(child_l, child_i, data);
//       }
//       // }
//     }
//   }

//  public:
//   support_refinement_recursive(int64_t dim, int64_t max_level, int64_t min_support,
//                      const std::vector<double> &data)
//       : dim(dim), max_level(max_level), min_support(min_support), data(data), num_visited(0) {
//     entries = data.size() / dim;
//   }

//   void refine() {
//     levels.clear();
//     indices.clear();
//     std::vector<int64_t> l(dim, 1);
//     std::vector<int64_t> i(dim, 1);
//     refine_impl(l, i, data);
//   }

//   std::vector<std::vector<int64_t>> &get_levels() { return levels; }
//   std::vector<std::vector<int64_t>> &get_indices() { return indices; }

//   void print_level(std::vector<int64_t> l) {
//     std::cout << "l: (";
//     for (int64_t d = 0; d < dim; d += 1) {
//       if (d > 0) {
//         std::cout << ", ";
//       }
//       std::cout << l[d];
//     }
//     std::cout << ")" << std::endl;
//   }
//   void print_index(std::vector<int64_t> i) {
//     std::cout << "i: (";
//     for (int64_t d = 0; d < dim; d += 1) {
//       if (d > 0) {
//         std::cout << ", ";
//       }
//       std::cout << i[d];
//     }
//     std::cout << ")" << std::endl;
//   }
//   void print_pos(std::vector<double> pos) {
//     std::cout << "pos: (";
//     for (int64_t d = 0; d < dim; d += 1) {
//       if (d > 0) {
//         std::cout << ", ";
//       }
//       std::cout << pos[d];
//     }
//     std::cout << ")" << std::endl;
//   }
// };

// void to_pos(const std::vector<int64_t> &l, const std::vector<int64_t> &i,
//             std::vector<double> &pos) {
//   int64_t dim = l.size();
//   for (int64_t d = 0; d < dim; d += 1) {
//     pos[d] = std::pow(2.0, static_cast<double>(-l[d])) * static_cast<double>(i[d]);
//   }
// }

// void write_grid_positions(const std::string &grid_file_name, const std::string
// &dataset_file_name,
//                           std::vector<std::vector<int64_t>> &ls,
//                           std::vector<std::vector<int64_t>> &is) {
//   std::ofstream out_file(grid_file_name);

//   int64_t dim = ls[0].size();
//   for (int64_t d = 0; d < dim; d += 1) {
//     if (d > 0) {
//       out_file << ", ";
//     }
//     out_file << "x" << std::to_string(d);
//   }
//   out_file << std::endl;
//   out_file << "#filename: " << dataset_file_name << std::endl;
//   std::vector<double> pos(dim);
//   for (int64_t i = 0; i < static_cast<int64_t>(ls.size()); i += 1) {
//     to_pos(ls[i], is[i], pos);
//     for (int64_t d = 0; d < dim; d += 1) {
//       if (d > 0) {
//         out_file << ", ";
//       }
//       out_file << pos[d];
//     }
//     out_file << std::endl;
//   }
// }

int main(void) {
  // csv_reader r("ripley.arff");
  // csv_reader r("gaussian_c2_size1000_dim2.arff");
  // csv_reader r("gaussian_c5_size1000_dim2.arff");
  std::string dataset_file_name("../datasets/gaussian/gaussian_c5_size1000_dim2_noise.arff");
  // csv_reader r(dataset_file_name);
  // r.print_data();
  // r.print_targets();
  // int64_t dim = r.get_dim();
  // int64_t entries = r.get_entries();

  sgpp::datadriven::Dataset dataset = sgpp::datadriven::ARFFTools::readARFF(dataset_file_name);
  int64_t entries = dataset.getNumberInstances();
  int64_t dim = dataset.getDimension();
  std::vector<double> data = std::move(dataset.getData());  // this won't actually move
  // csv_reader r("../datasets/gaussian/gaussian_c100_size1000000_dim10.arff");

  std::cout << "dim: " << dim << " entries: " << entries << std::endl;
  // std::vector<double> &data = r.get_data();
  // std::vector<double> &targets = r.get_targets();

  int64_t max_level = 7;
  int64_t min_support = 50;
  std::string grid_file_name("refined_grid.csv");
  // level_generator l_gen(dim, level);
  // while (l_gen.next()) {
  //   std::cout << "level: ";
  //   for (int64_t d = 0; d < dim; d += 1) {
  //     if (d > 0) {
  //       std::cout << ", ";
  //     }
  //     std::cout << l_gen.get(d);
  //   }
  //   index_generator i_gen(dim, l_gen.get());
  //   std::cout << " indices: ";
  //   bool first = true;
  //   while (i_gen.next()) {
  //     if (first) {
  //       first = false;
  //     } else {
  //       std::cout << ", ";
  //     }
  //     std::cout << "(";
  //     for (int64_t d = 0; d < dim; d += 1) {
  //       if (d > 0) {
  //         std::cout << ", ";
  //       }
  //       std::cout << l_gen.get(d);
  //     }
  //     std::cout << ")";
  //   }
  //   std::cout << std::endl;
  // }
  sgpp::datadriven::support_refinement_recursive ref(dim, max_level, min_support, data);
  ref.refine();

  std::vector<std::vector<int64_t>> &ls = ref.get_levels();
  std::vector<std::vector<int64_t>> &is = ref.get_indices();
  // for (int64_t gp = 0; gp < static_cast<int64_t>(ls.size()); gp += 1) {
  //   std::cout << "l: (";
  //   for (int64_t d = 0; d < dim; d += 1) {
  //     if (d > 0) {
  //       std::cout << ", ";
  //     }
  //     std::cout << ls[gp][d];
  //   }
  //   std::cout << "), i: (";
  //   for (int64_t d = 0; d < dim; d += 1) {
  //     if (d > 0) {
  //       std::cout << ", ";
  //     }
  //     std::cout << is[gp][d];
  //   }
  //   std::cout << ")" << std::endl;
  // }

  std::unique_ptr<sgpp::base::Grid> grid =
      std::unique_ptr<sgpp::base::Grid>(sgpp::base::Grid::createLinearGrid(dim));
  // sgpp::base::GridGenerator &grid_generator = grid->getGenerator();
  sgpp::base::GridStorage &grid_storage = grid->getStorage();
  sgpp::base::HashGridPoint p(dim);
  for (int64_t gp = 0; gp < static_cast<int64_t>(ls.size()); gp += 1) {
    for (int64_t d = 0; d < dim; d += 1) {
      p.set(d, ls[gp][d], is[gp][d]);
    }
    grid_storage.insert(p);
  }

  if (ls.size() > 0) {
    std::cout << "final number of grid points created: " << (ls.size() / dim) << std::endl;
    sgpp::datadriven::support_refinement_recursive::write_grid_positions(grid_file_name,
                                                                         dataset_file_name, ls, is);
  } else {
    std::cerr << "error: did not create any grid points" << std::endl;
  }
  return 1;
}
