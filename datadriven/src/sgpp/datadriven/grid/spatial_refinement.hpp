#pragma once

namespace sgpp {
namespace datadriven {

class spatial_refinement {
 private:
  int64_t dim;
  int64_t max_level;
  int64_t min_support;
  std::vector<double> data;

  int64_t entries;
  std::vector<std::vector<int64_t>> levels;
  std::vector<std::vector<int64_t>> indices;
  int64_t num_visited;

  std::vector<double> bound_left;
  std::vector<double> bound_right;

  std::vector<int64_t> child_l;
  std::vector<int64_t> child_i;

  int64_t regular_accept_level = 1;

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
                   size_t data_index, size_t changed_dim) {
    if (data[changed_dim * entries + data_index] < bound_left[changed_dim] ||
        data[changed_dim * entries + data_index] > bound_right[changed_dim]) {
      return false;
    }
    return true;
  }

  std::vector<size_t> __attribute__((noinline))
  filter(const std::vector<size_t> &cur_indices, std::vector<int64_t> &cur_l,
         std::vector<int64_t> &cur_i, size_t changed_dim) {
    std::vector<size_t> filtered_indices;
    filtered_indices.reserve(cur_indices.size());
    for (size_t i : cur_indices) {
      if (has_support(bound_left, bound_right, i, changed_dim)) {
        filtered_indices.push_back(i);
      }
    }
    return filtered_indices;
  }

  int64_t __attribute__((noinline))
  verify_support(const std::vector<size_t> &cur_indices, std::vector<int64_t> &cur_l,
                 std::vector<int64_t> &cur_i, size_t changed_dim) {
    int64_t num_support = 0;
    for (size_t i : cur_indices) {
      if (has_support(bound_left, bound_right, i, changed_dim)) {
        num_support += 1;
      }
    }
    return num_support;
  }

  // N * m data points accessed (~memory operations)
  void refine_impl(std::vector<int64_t> cur_l, std::vector<int64_t> cur_i,
                   const std::vector<size_t> &cur_indices, int64_t sum_l, size_t changed_dim) {
    // already visited?
    num_visited += 1;
    if (num_visited % 1000 == 0) {
      std::cout << "num_visited: " << num_visited << std::endl;
    }

    neighbor_pos_dir(cur_l, cur_i, bound_left, direction::left);
    neighbor_pos_dir(cur_l, cur_i, bound_right, direction::right);

    // print_level(cur_l);
    // print_index(cur_i);
    // print_pos(bound_left);
    // print_pos(bound_right);

    bool use_filtered = false;
    std::vector<size_t> filtered_indices;
    if (sum_l > regular_accept_level + dim - 1) {
      // std::cout << "slow point" << std::endl;
      // int64_t cur_entries = data.size() / dim;
      use_filtered = true;
      filtered_indices = filter(cur_indices, cur_l, cur_i, changed_dim);
      int64_t num_support = filtered_indices.size();

      // std::cout << "num_support: " << num_support << std::endl;
      if (num_support < min_support) {
        return;
      }

      // std::cout << "filtered_indices.size(): " << filtered_indices.size()
      //           << " cur_indices.size(): " << cur_indices.size()
      //           << " diff: " << (cur_indices.size() - filtered_indices.size()) << std::endl;
    }

    levels.push_back(cur_l);
    indices.push_back(cur_i);

    if (levels.size() % 100 == 0) {
      std::cout << "reached size: " << levels.size() << std::endl;
    }
    // recurse children if lower dims are all 1 (but not necessarily the
    // current dim)
    if (sum_l < max_level + dim - 1) {
      for (int64_t d = 0; d < dim; d += 1) {
        child_d_dir(cur_l, cur_i, child_l, child_i, d, direction::left);
        if (use_filtered) {
          refine_impl(child_l, child_i, filtered_indices, sum_l + 1, d);
        } else {
          refine_impl(child_l, child_i, cur_indices, sum_l + 1, d);
        }
        child_d_dir(cur_l, cur_i, child_l, child_i, d, direction::right);
        if (use_filtered) {
          refine_impl(child_l, child_i, filtered_indices, sum_l + 1, d);
        } else {
          refine_impl(child_l, child_i, cur_indices, sum_l + 1, d);
        }
        if (cur_l[d] != 1) {
          break;
        }
      }
    }
  }

 public:
  spatial_refinement(int64_t dim, int64_t max_level, int64_t min_support,
                     const std::vector<double> &data)
      : dim(dim),
        max_level(max_level),
        min_support(min_support),
        num_visited(0),
        bound_left(dim),
        bound_right(dim),
        child_l(dim),
        child_i(dim) {
    entries = data.size() / dim;
    this->data.resize(data.size());
    // convert data to SoA
    for (int64_t i = 0; i < entries; i += 1) {
      for (int64_t d = 0; d < dim; d += 1) {
        this->data[d * entries + i] = data[i * dim + d];
      }
    }
  }

  void refine() {
    levels.clear();
    indices.clear();
    std::vector<size_t> unfiltered_indices(entries);
    for (size_t i = 0; i < unfiltered_indices.size(); i += 1) {
      unfiltered_indices[i] = i;
    }
    std::vector<int64_t> l(dim, 1);
    std::vector<int64_t> i(dim, 1);
    refine_impl(l, i, unfiltered_indices, dim, 0);
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
