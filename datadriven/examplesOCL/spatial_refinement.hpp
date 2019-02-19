#pragma once

namespace sgpp {
namespace datadriven {

class spatial_refinement {
 private:
  int64_t dim;
  int64_t max_level;
  int64_t min_support;
  const std::vector<double> &data;

  int64_t entries;
  std::vector<std::vector<int64_t>> levels;
  std::vector<std::vector<int64_t>> indices;
  std::set<std::pair<std::vector<int64_t>, std::vector<int64_t>>> visited;
  int64_t num_visited;

  enum class direction { left, right };

  void child_d_dir(std::vector<int64_t> &l, std::vector<int64_t> &i, std::vector<int64_t> &child_l,
                   std::vector<int64_t> &child_i, int64_t d, direction dir) {
    child_l = l;
    child_i = i;
    child_l[d] += 1;
    if (dir == direction::left) {
      child_i[d] = 2 * child_i[d] - 1;
    } else if (dir == direction::right) {
      child_i[d] = 2 * child_i[d] + 1;
    } else {
      throw;  // never
    }
  }

  void neighbor_pos_dir(std::vector<int64_t> &l, std::vector<int64_t> &i,
                        std::vector<double> &neighbor, direction dir) {
    if (dir == direction::left) {
      for (int64_t d = 0; d < dim; d += 1) {
        neighbor[d] = std::pow(2.0, static_cast<double>(-l[d])) * static_cast<double>(i[d] - 1ll);
      }
    } else if (dir == direction::right) {
      for (int64_t d = 0; d < dim; d += 1) {
        neighbor[d] = std::pow(2.0, static_cast<double>(-l[d])) * static_cast<double>(i[d] + 1ll);
      }
    } else {
      throw;  // never
    }
  }

  bool has_support(std::vector<double> &bound_left, std::vector<double> &bound_right,
                   std::vector<double> &data_point) {
    for (int64_t d = 0; d < dim; d += 1) {
      if (data_point[d] < bound_left[d] || data_point[d] > bound_right[d]) {
        return false;
      }
    }
    return true;
  }

  // N * m data points accessed (~memory operations)
  void refine_impl(std::vector<int64_t> cur_l, std::vector<int64_t> cur_i,
                   const std::vector<double> &data) {
    // already visited?
    num_visited += 1;
    if (num_visited % 1000 == 0) {
      std::cout << "num_visited: " << num_visited << std::endl;
    }

    auto p = std::make_pair(cur_l, cur_i);
    if (visited.find(std::make_pair(cur_l, cur_i)) != visited.end()) {
      return;
    }
    visited.insert(p);

    // max_level reached?
    int64_t sum_l = 0;
    for (int64_t d = 0; d < dim; d += 1) {
      sum_l += cur_l[d];
    }
    if (sum_l > max_level + dim - 1) {
      return;
    }
    // validate that the current grid point has enough grid points on its
    // support
    std::vector<double> bound_left(dim);
    neighbor_pos_dir(cur_l, cur_i, bound_left, direction::left);
    std::vector<double> bound_right(dim);
    neighbor_pos_dir(cur_l, cur_i, bound_right, direction::right);
    std::vector<double> data_point(dim);
    // std::cout << "---------- l, i, n_left, n_right, data points -----------"
    //           << std::endl;
    // print_level(cur_l);
    // print_index(cur_i);
    // print_pos(bound_left);
    // print_pos(bound_right);
    int64_t num_support = 0;
    int64_t cur_entries = data.size() / dim;
    for (int64_t i = 0; i < cur_entries; i += 1) {
      for (int64_t d = 0; d < dim; d += 1) {
        data_point[d] = data[i * dim + d];
      }
      if (has_support(bound_left, bound_right, data_point)) {
        num_support += 1;
      }
    }
    // std::cout << "num_support: " << num_support << std::endl;
    if (num_support < min_support) {
      return;
    }

    std::vector<double> filtered_dataset;
    bool use_filtered = false;
    if (num_support < 0.5 * data.size()) {
      use_filtered = true;
      for (int64_t i = 0; i < cur_entries; i += 1) {
        for (int64_t d = 0; d < dim; d += 1) {
          data_point[d] = data[i * dim + d];
        }
        if (has_support(bound_left, bound_right, data_point)) {
          for (int64_t d = 0; d < dim; d += 1) {
            filtered_dataset.push_back(data[i * dim + d]);
          }
        }
      }
    }

    levels.push_back(cur_l);
    indices.push_back(cur_i);

    if (levels.size() % 100 == 0) {
      std::cout << "reached size: " << levels.size() << std::endl;
    }
    // recurse children
    // TODO: need strategy to avoid duplicate recursive calls (or simply use
    // hashmap)
    if (sum_l < max_level + dim - 1) {
      std::vector<int64_t> child_l(dim);
      std::vector<int64_t> child_i(dim);
      for (int64_t d = 0; d < dim; d += 1) {
        child_d_dir(cur_l, cur_i, child_l, child_i, d, direction::left);
        if (use_filtered) {
          refine_impl(child_l, child_i, filtered_dataset);
        } else {
          refine_impl(child_l, child_i, data);
        }
        child_d_dir(cur_l, cur_i, child_l, child_i, d, direction::right);
        if (use_filtered) {
          refine_impl(child_l, child_i, filtered_dataset);
        } else {
          refine_impl(child_l, child_i, data);
        }
      }
    }
  }

 public:
  spatial_refinement(int64_t dim, int64_t max_level, int64_t min_support,
                     const std::vector<double> &data)
      : dim(dim), max_level(max_level), min_support(min_support), data(data), num_visited(0) {
    entries = data.size() / dim;
  }

  void refine() {
    levels.clear();
    indices.clear();
    std::vector<int64_t> l(dim, 1);
    std::vector<int64_t> i(dim, 1);
    refine_impl(l, i, data);
  }

  std::vector<std::vector<int64_t>> &get_levels() { return levels; }
  std::vector<std::vector<int64_t>> &get_indices() { return indices; }

  void print_level(std::vector<int64_t> l) {
    std::cout << "l: (";
    for (int64_t d = 0; d < dim; d += 1) {
      if (d > 0) {
        std::cout << ", ";
      }
      std::cout << l[d];
    }
    std::cout << ")" << std::endl;
  }
  void print_index(std::vector<int64_t> i) {
    std::cout << "i: (";
    for (int64_t d = 0; d < dim; d += 1) {
      if (d > 0) {
        std::cout << ", ";
      }
      std::cout << i[d];
    }
    std::cout << ")" << std::endl;
  }
  void print_pos(std::vector<double> pos) {
    std::cout << "pos: (";
    for (int64_t d = 0; d < dim; d += 1) {
      if (d > 0) {
        std::cout << ", ";
      }
      std::cout << pos[d];
    }
    std::cout << ")" << std::endl;
  }

  static void to_pos(const std::vector<int64_t> &l, const std::vector<int64_t> &i,
                     std::vector<double> &pos) {
    int64_t dim = l.size();
    for (int64_t d = 0; d < dim; d += 1) {
      pos[d] = std::pow(2.0, static_cast<double>(-l[d])) * static_cast<double>(i[d]);
    }
  }

  static void write_grid_positions(const std::string &grid_file_name,
                                   const std::string &dataset_file_name,
                                   std::vector<std::vector<int64_t>> &ls,
                                   std::vector<std::vector<int64_t>> &is) {
    std::ofstream out_file(grid_file_name);

    int64_t dim = ls[0].size();
    for (int64_t d = 0; d < dim; d += 1) {
      if (d > 0) {
        out_file << ", ";
      }
      out_file << "x" << std::to_string(d);
    }
    out_file << std::endl;
    out_file << "#filename: " << dataset_file_name << std::endl;
    std::vector<double> pos(dim);
    for (int64_t i = 0; i < static_cast<int64_t>(ls.size()); i += 1) {
      to_pos(ls[i], is[i], pos);
      for (int64_t d = 0; d < dim; d += 1) {
        if (d > 0) {
          out_file << ", ";
        }
        out_file << pos[d];
      }
      out_file << std::endl;
    }
  }
};

}  // namespace datadriven
}  // namespace sgpp
