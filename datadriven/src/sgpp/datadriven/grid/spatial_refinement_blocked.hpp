#pragma once

#include <cinttypes>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace sgpp::datadriven {

class spatial_refinement_blocked {
 private:
  int64_t dim;
  int64_t max_level;
  int64_t min_support;
  std::vector<double> data;

  int64_t entries;
  std::vector<int64_t> levels;
  std::vector<int64_t> indices;
  int64_t num_visited;

  std::vector<int64_t> schedule_level;
  std::vector<int64_t> schedule_index;
  std::vector<bool> schedule_support;

  std::vector<int64_t> schedule_next_level;
  std::vector<int64_t> schedule_next_index;

  enum class direction { left, right };

#if USE_OCL == 1
  bool use_ocl = false;
#endif

  inline void child_d_dir(std::vector<int64_t> &l, std::vector<int64_t> &i,
                          std::vector<int64_t> &child_l, std::vector<int64_t> &child_i, int64_t d,
                          direction dir) {
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

  inline void neighbor_pos_dir(std::vector<int64_t> &l, std::vector<int64_t> &i,
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

  void neighbor_pos_dir_blocked(std::vector<int64_t> &schedule_level,
                                std::vector<int64_t> &schedule_index,
                                std::vector<double> &neighbors, direction dir);

  void verify_support_blocked();

#if USE_OCL == 1
  void verify_support_blocked_OCL();
#endif

  void refine_impl();

 public:
  spatial_refinement_blocked(int64_t dim, int64_t max_level, int64_t min_support,
                             const std::vector<double> &data);
  void refine();
  std::vector<int64_t> &get_levels();
  std::vector<int64_t> &get_indices();
  void print_level(std::vector<int64_t> l);
  void print_index(std::vector<int64_t> i);
  void print_pos(std::vector<double> pos);
  static void write_grid_positions(int64_t dim, const std::string &grid_file_name,
                                   const std::string &dataset_file_name, std::vector<int64_t> &ls,
                                   std::vector<int64_t> &is);

#if USE_OCL == 1
  void set_OCL(bool use_ocl = true) { this->use_ocl = use_ocl; }
#endif

};  // namespace datadriven

}  // namespace sgpp::datadriven
